import argparse
import gc
import glob
import logging
import multiprocessing as mp
import os
import subprocess
import sys

import json5
import torch
import torch.distributed as dist
from PIL import Image

import wan
from api_client import generate_video, get_api_client
from extract_keyframes import save_keyframes
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Story-to-Video Pipeline with API support"
    )
    parser.add_argument("--story_script_path", type=str, default="./story/neo.json")

    # 模型路径
    parser.add_argument(
        "--t2v_model_path", type=str, default="/path/to/Wan2.2-T2V-A14B"
    )
    parser.add_argument(
        "--i2v_model_path", type=str, default="/path/to/Wan2.2-I2V-A14B"
    )

    # API 配置
    parser.add_argument(
        "--use_api",
        action="store_true",
        default=False,
        help="Use API for first shot generation",
    )
    parser.add_argument(
        "--api_provider", type=str, default="wan", choices=["wan"], help="API provider"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (or use env DASHSCOPE_API_KEY)",
    )

    # 视频配置
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--max_memory_size", type=int, default=8)
    parser.add_argument("--input_dir", type=str, default="./input")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.txt")

    # 推理配置
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"]
    )
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=3.5)
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument(
        "--offload_model", action="store_true", help="Offload model to CPU"
    )

    # 模式选择
    parser.add_argument(
        "--t2v_first_shot", action="store_true", help="Use T2V model for first shot"
    )
    parser.add_argument(
        "--m2v_first_shot",
        action="store_true",
        help="Use M2V model for first shot (requires image)",
    )
    parser.add_argument(
        "--mi2v",
        action="store_true",
        help="Start from last frame of last video with MI2V",
    )
    parser.add_argument(
        "--mm2v",
        action="store_true",
        help="Start from last 5 frames of last video with MM2V",
    )
    parser.add_argument("--fix", type=int, default=3, help="Fix the first n keyframes")

    # LoRA 配置
    parser.add_argument("--finetune_checkpoint_dir", type=str, default=None)
    parser.add_argument("--lora_weight_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)

    args = parser.parse_args()
    return args


def _init_logging(rank, log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout),
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.ERROR, handlers=[logging.StreamHandler(stream=sys.stdout)]
        )


def generate_first_shot_api(prompt: str, output_path: str, size: str, **kwargs):
    """
    使用 API 生成第一个镜头

    Args:
        prompt: 文本描述
        output_path: 输出路径
        size: 分辨率 (如 "832*480")
    """
    logging.info(f"Generating first shot with API: {prompt[:50]}...")

    # 尝试使用 API 生成
    video_path = generate_video(
        prompt=prompt,
        mode="t2v",
        size=size,
        provider=kwargs.get("api_provider", "wan"),
        save_path=output_path,
        api_key=kwargs.get("api_key"),
    )

    logging.info(f"API video saved to: {video_path}")
    return video_path


def generate_first_shot_local(
    prompt: str, output_path: str, args, t2v_config, size, rank: int = 0
):
    """使用本地模型生成第一个镜头 (T2V)"""
    logging.info("Loading T2V model...")
    t2v_model = wan.WanT2V(
        config=t2v_config,
        checkpoint_dir=args.t2v_model_path,
        device_id=rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    video = t2v_model.generate(
        prompt,
        size=SIZE_CONFIGS[size],
        frame_num=t2v_config.frame_num,
        shift=t2v_config.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=t2v_config.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.seed,
        offload_model=args.offload_model,
    )

    if rank == 0:
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=t2v_config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logging.info(f"Local T2V video saved to: {output_path}")

    del t2v_model
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    # 初始化
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank, args.log_file)

    # 检查 API 模式
    if args.use_api and rank == 0:
        logging.info("=" * 50)
        logging.info("API MODE ENABLED - First shot will use API")
        logging.info("=" * 50)

    # 分布式配置
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )

    if dist.is_initialized():
        base_seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.seed = base_seed[0]

    # 加载剧本
    story_script = json5.load(open(args.story_script_path, "r", encoding="utf-8"))
    os.makedirs(args.output_dir, exist_ok=True)

    first_shot_output = f"{args.output_dir}/01_01.mp4"

    # 生成第一个镜头
    first_prompt = story_script["scenes"][0]["video_prompts"][0]

    if args.t2v_first_shot:
        # 本地 T2V 模型
        t2v_config = WAN_CONFIGS["t2v-A14B"]
        if rank == 0:
            generate_first_shot_local(
                first_prompt, first_shot_output, args, t2v_config, args.size, rank
            )
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    elif args.use_api and rank == 0:
        # API 模式
        generate_first_shot_api(
            prompt=first_prompt,
            output_path=first_shot_output,
            size=args.size,
            api_provider=args.api_provider,
            api_key=args.api_key,
        )
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()

    elif not os.path.exists(first_shot_output) and rank == 0:
        # 如果没有第一个镜头且没有指定模式，询问用户或使用 API
        logging.warning(f"First shot not found: {first_shot_output}")
        logging.info("Please generate the first shot using API or local model.")
        logging.info("You can use: python pipeline.py --use_api")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # 提取第一个镜头的关键帧
    if rank == 0:
        if os.path.exists(first_shot_output):
            save_keyframes(first_shot_output)
        else:
            logging.warning(f"First shot not found, skipping keyframe extraction")

    # 同步
    if world_size > 1:
        dist.barrier()

    # 加载 M2V 模型
    m2v_config = WAN_CONFIGS["m2v-A14B"]
    if args.lora_weight_path is not None:
        m2v_config.low_noise_lora.weight = os.path.join(
            args.lora_weight_path, "backbone_low_noise.pth"
        )
        m2v_config.high_noise_lora.weight = os.path.join(
            args.lora_weight_path, "backbone_high_noise.pth"
        )
    if args.lora_rank is not None:
        m2v_config.low_noise_lora.r = m2v_config.low_noise_lora.lora_alpha = (
            args.lora_rank
        )
        m2v_config.high_noise_lora.r = m2v_config.high_noise_lora.lora_alpha = (
            args.lora_rank
        )

    logging.info("Loading M2V model...")
    m2v_model = wan.WanM2V(
        config=m2v_config,
        checkpoint_dir=args.i2v_model_path,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        finetune_checkpoint_dir=args.finetune_checkpoint_dir,
    )

    # 生成后续镜头
    for scene in story_script["scenes"]:
        scene_num = scene["scene_num"]

        for i, prompt in enumerate(scene["video_prompts"]):
            shot_num = i + 1

            # 跳过第一个镜头（已生成）
            if scene_num == 1 and shot_num == 1:
                continue

            logging.info(f"Generating Scene {scene_num} / Shot {shot_num}: {prompt}")

            # 获取记忆银行
            memory_bank = sorted(glob.glob(f"{args.output_dir}/*keyframe*.jpg"))
            if len(memory_bank) > args.max_memory_size:
                memory_bank = (
                    memory_bank[: args.fix]
                    + memory_bank[-(args.max_memory_size - args.fix) :]
                )

            # 检查是否需要 MI2V/MM2V
            first_frame_file = None
            motion_frames_file = None

            if args.mi2v and not scene.get("cut", [True])[i]:
                first_frame_file = f"{args.output_dir}/last_frame.jpg"
            if args.mm2v and not scene.get("cut", [True])[i]:
                motion_frames_file = f"{args.output_dir}/motion_frames.mp4"

            # 生成视频
            video = m2v_model.generate(
                prompt,
                memory_bank,
                first_frame_file=first_frame_file,
                motion_frames_file=motion_frames_file,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=m2v_config.frame_num,
                shift=m2v_config.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=m2v_config.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.seed + i,
                offload_model=args.offload_model,
            )

            if rank == 0:
                output_file = f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4"

                # 裁剪
                if first_frame_file is not None:
                    video = video[:, 1:]
                elif motion_frames_file is not None:
                    video = video[:, 5:]

                save_video(
                    tensor=video[None],
                    save_file=output_file,
                    fps=m2v_config.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

                # 提取关键帧
                save_keyframes(output_file)
                logging.info(f"Saved: {output_file}")

                del video
                torch.cuda.empty_cache()

            if world_size > 1:
                dist.barrier()

    # 拼接所有视频
    if rank == 0:
        videos = sorted(glob.glob(f"{args.output_dir}/*.mp4"))
        list_path = os.path.join(args.output_dir, "concat_list.txt")

        with open(list_path, "w", encoding="utf-8") as f:
            for v in videos:
                f.write(f"file '{os.path.abspath(v)}'\n")

        out = os.path.join(args.output_dir, f"{os.path.basename(args.output_dir)}.mp4")

        # 尝试直接拼接
        ret = subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                "-y",
                out,
            ]
        )

        # 如果失败，重新编码
        if ret.returncode != 0:
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_path,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-r",
                    "30",
                    "-y",
                    out,
                ],
                check=True,
            )

        logging.info(f"Final video: {out}")

    # 清理
    torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished!")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
