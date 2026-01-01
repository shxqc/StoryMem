# API Client for Video Generation
# 支持多种视频生成 API
import base64
import os
import time
from abc import ABC, abstractmethod
from io import BytesIO

import requests
from PIL import Image

class BaseVideoAPI(ABC):
    """视频生成 API 基类"""

    @abstractmethod
    def generate_t2v(self, prompt: str, **kwargs):
        """文本生成视频"""
        pass

    @abstractmethod
    def generate_i2v(self, prompt: str, image: Image.Image, **kwargs):
        """图像生成视频"""
        pass


class WanAPIClient(BaseVideoAPI):
    """
    阿里云 Wan API 客户端
    文档: https://help.aliyun.com/zh/dashscope/developer-reference/video-generation
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation"
        )

    def _call_api(self, model: str, input_data: dict, **kwargs) -> dict:
        """调用 API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "input": input_data, "parameters": kwargs}

        response = requests.post(
            self.base_url, json=payload, headers=headers, timeout=300
        )
        response.raise_for_status()
        return response.json()

    def generate_t2v(
        self, prompt: str, size: str = "832*480", duration: str = "5", **kwargs
    ) -> str:
        """
        文本生成视频

        Args:
            prompt: 文本描述
            size: 分辨率，如 "832*480"
            duration: 时长，如 "5" 秒

        Returns:
            生成的视频 URL
        """
        width, height = size.split("*")
        input_data = {
            "prompt": prompt,
            "size": f"{width}x{height}",
            "duration": duration,
        }

        result = self._call_api("wan-v2.1-t2v", input_data, **kwargs)
        task_id = result.get("output", {}).get("task_id")

        # 轮询结果
        return self._wait_for_result(task_id)

    def generate_i2v(
        self,
        prompt: str,
        image: Image.Image,
        size: str = "832*480",
        duration: str = "5",
        **kwargs,
    ) -> str:
        """
        图像生成视频

        Args:
            prompt: 文本描述
            image: 输入图像
            size: 分辨率
            duration: 时长

        Returns:
            生成的视频 URL
        """
        width, height = size.split("*")

        # 图片转 base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        input_data = {
            "prompt": prompt,
            "image": f"data:image/png;base64,{img_base64}",
            "size": f"{width}x{height}",
            "duration": duration,
        }

        result = self._call_api("wan-v2.1-i2v", input_data, **kwargs)
        task_id = result.get("output", {}).get("task_id")

        return self._wait_for_result(task_id)

    def _wait_for_result(self, task_id: str, poll_interval: int = 5) -> str:
        """轮询获取结果"""
        status_url = f"{self.base_url}/tasks/{task_id}"

        while True:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(status_url, headers=headers)
            response.raise_for_status()
            result = response.json()

            status = result.get("output", {}).get("task_status")
            if status == "SUCCEEDED":
                return result.get("output", {}).get("video_url")
            elif status in ["FAILED", "CANCELLED"]:
                raise RuntimeError(f"Task failed: {result}")
            else:
                print(f"Status: {status}, waiting...")
                time.sleep(poll_interval)

    def download_video(self, video_url: str, save_path: str):
        """下载视频到本地"""
        response = requests.get(video_url, timeout=300)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)


class ReplicateAPI(BaseVideoAPI):
    """
    Replicate API 客户端
    用于调用开源视频生成模型
    """

    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        self.base_url = "https://api.replicate.com/v1"

    def _call_api(self, owner: str, model: str, version: str, input_data: dict) -> str:
        """调用 Replicate API"""
        url = f"{self.base_url}/models/{owner}/{model}/versions/{version}"
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json",
        }

        # 创建任务
        response = requests.post(url, json={"input": input_data}, headers=headers)
        response.raise_for_status()
        task_url = response.json().get("urls", {}).get("get")

        # 轮询结果
        return self._wait_for_result(task_url)

    def _wait_for_result(self, task_url: str, poll_interval: int = 2) -> str:
        """轮询获取结果"""
        headers = {"Authorization": f"Token {self.api_token}"}

        while True:
            response = requests.get(task_url, headers=headers)
            response.raise_for_status()
            result = response.json()

            status = result.get("status")
            if status == "succeeded":
                return result.get("output")
            elif status in ["failed", "canceled"]:
                raise RuntimeError(f"Task failed: {result}")
            else:
                print(f"Status: {status}, waiting...")
                time.sleep(poll_interval)

    def generate_t2v(self, prompt: str, **kwargs) -> str:
        """文本生成视频 (示例使用 ModelScope)"""
        input_data = {
            "prompt": prompt,
            "num_frames": kwargs.get("num_frames", 81),
            "width": kwargs.get("width", 832),
            "height": kwargs.get("height", 480),
        }
        return self._call_api(
            "modelscope", "stable-video-diffusion", "3f036d649b7fcf5d0", input_data
        )

    def generate_i2v(self, prompt: str, image: Image.Image, **kwargs) -> str:
        """图像生成视频"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        input_data = {
            "prompt": prompt,
            "input_image": f"data:image/png;base64,{img_base64}",
            "num_frames": kwargs.get("num_frames", 81),
        }
        return self._call_api(
            "stability-ai",
            "stable-video-diffusion-img2vid",
            "3f1d4c4c7c0c36",
            input_data,
        )


def get_api_client(provider: str = "wan", **kwargs) -> BaseVideoAPI:
    """
    获取 API 客户端

    Args:
        provider: 提供商 ("wan", "replicate")
        **kwargs: 其他参数

    Returns:
        API 客户端实例
    """
    if provider == "wan":
        return WanAPIClient(**kwargs)
    elif provider == "replicate":
        return ReplicateAPI(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# 便捷函数
def generate_video(
    prompt: str,
    mode: str = "t2v",
    image: Image.Image = None,
    size: str = "832*480",
    provider: str = "wan",
    save_path: str = None,
    **kwargs,
):
    """
    便捷的视频生成函数

    Args:
        prompt: 文本描述
        mode: 模式 ("t2v", "i2v")
        image: 输入图像（i2v 模式需要）
        size: 分辨率
        provider: API 提供商
        save_path: 保存路径（可选）
        **kwargs: 其他参数

    Returns:
        视频文件路径
    """
    client = get_api_client(provider)
    duration = kwargs.pop("duration", "5")

    if mode == "t2v":
        video_url = client.generate_t2v(prompt, size=size, duration=duration, **kwargs)
    elif mode == "i2v":
        if image is None:
            raise ValueError("i2v mode requires image parameter")
        video_url = client.generate_i2v(
            prompt, image, size=size, duration=duration, **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 下载到本地
    if save_path is None:
        save_path = f"temp_video_{int(time.time())}.mp4"

    client.download_video(video_url, save_path)
    return save_path


if __name__ == "__main__":
    # 测试
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--save", type=str, default="test_output.mp4")
    args = parser.parse_args()

    image = None
    if args.image:
        image = Image.open(args.image)

    video_path = generate_video(
        args.prompt, mode=args.mode, image=image, size=args.size, save_path=args.save
        size=args.size,
        save_path=args.save
    )
    print(f"Video saved to: {video_path}")
