import importlib.metadata
import io
import json
import numpy as np
import subprocess

from datetime import datetime
from pathlib import Path
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoTokenizer
from typing import Dict, Optional
from urllib.parse import urlparse, unquote

from vllm.assets.audio import AudioAsset

from llmperf.config.models import Model
from llmperf.config.workloads import Request
from llmperf.constants import ALL_STRATEGY_PARAMS
from llmperf.promptpreparation.video import VideoAsset

def get_version(package_name: str) -> str:
    try:
        version = importlib.metadata.version(package_name)
        
        dist = importlib.metadata.distribution(package_name)
        dist_name = dist.metadata["Name"].replace("-", "_")
        dist_info_dir = Path(dist.locate_file("")).joinpath(f"{dist_name}-{version}.dist-info")
        direct_url_path = dist_info_dir / "direct_url.json"
        
        if direct_url_path.exists():
            with open(direct_url_path) as f:
                data = json.load(f)
                vcs_info = data.get("vcs_info", {})
                if "commit_id" in vcs_info:
                    return vcs_info.get("requested_revision") or vcs_info["commit_id"]

                url = data.get("url", "")
                parsed = urlparse(url)
                if parsed.scheme == "file":
                    path = Path(unquote(parsed.path))
                    if (path / ".git").exists():
                        try:
                            tag = subprocess.check_output(["git", "describe", "--tags"], cwd=path).decode().strip()
                            return tag
                        except subprocess.CalledProcessError:
                            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()[:9]
                            return commit

        return version
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def create_experiment_id(
    workload: str, model: str, approach: str,
    gpu_util: float, swap_space: int, num_gpu_blocks: Optional[int],
    max_model_len: Optional[int], max_num_batched_tokens: Optional[int],
    **kwargs
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version = get_version("vllm")

    max_model_len = max_model_len if max_model_len is not None else "def"
    max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else "def"
    num_gpu_blocks = num_gpu_blocks if num_gpu_blocks is not None else "def"

    parts = [
        workload,
        model,
        approach,
        version,
        timestamp,
        f"maxlen{max_model_len}",
        f"batch{max_num_batched_tokens}",
        f"blocks{num_gpu_blocks}",
        f"gpu{gpu_util}",
        f"swap{swap_space}",
    ]

    for key, value in kwargs.items():
        parts.append(f"{key}{value}")

    return "__".join(parts)

def load_image(path: str) -> ImageFile.ImageFile:
    return Image.open(path)

def load_image_from_array(array: np.ndarray) -> Image.Image:
    img = Image.fromarray(array.astype("uint8"))
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    image_file = Image.open(buf)
    return image_file

def duration_to_frames(duration, min_duration=1, max_duration=180, min_frames=8, max_frames=64):
    duration = max(min_duration, min(max_duration, duration))
    scale = (duration - min_duration) / (max_duration - min_duration)
    frames = min_frames + scale * (max_frames - min_frames)
    return int(frames)

def prepare_text_prompt(request: Request, model: Model) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model.path)
    if tokenizer.chat_template is None:
        tokenizer = AutoProcessor.from_pretrained(model.path, use_fast=True)
    
    if tokenizer.chat_template is None:
        return {
            "prompt": f"<|User|>: {request.input}\n\n<|Assistant|>:"
        }

    prompt = [
        {"role": "user", "content": request.input}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )
    
    final_prompt = {
        "prompt": formatted_prompt
    }
    return final_prompt

def prepare_image_prompt(request: Request, model: Model, compression_ratio: float = None) -> Dict:
    image = load_image(request.modality_path)

    if image.mode != "RGB" and model.alias.startswith("gemma-3"):
        image = image.convert("RGB")

    if compression_ratio:
        # Proportional Compression
        w, h = image.size
        if model.alias.startswith("qwen-2"):
            n_w = max(28, int(w - (compression_ratio * w)))
            n_h = max(28, int(h - (compression_ratio * h)))
        else:
            n_w = int(w - (compression_ratio * w))
            n_h = int(h - (compression_ratio * h))
        image = image.resize((n_w, n_h), resample=Image.Resampling.LANCZOS)

    processor = AutoProcessor.from_pretrained(model.path, use_fast=True)

    if processor.chat_template is None:
        return {
            "prompt": f"<|User|>: <image>\n{request.input}\n\n<|Assistant|>:",
            "multi_modal_data": {"image": image}
        }
    
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": request.input}
            ]
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )

    final_prompt = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image}
    }
    return final_prompt

def prepare_multi_image_prompt(request: Request, model: Model, num_frames: int = None, strategy: str = "uniform", smart_resize: bool = False) -> Dict:
    processor = AutoProcessor.from_pretrained(model.path, use_fast=True)

    num_available_frames = request.modality_size["frame_count"]
    if num_available_frames == 0:
        return {}

    if not num_frames:
        num_frames = duration_to_frames(request.modality_size["duration"])
    num_frames = num_available_frames if num_available_frames < num_frames else num_frames

    strategy_params = ALL_STRATEGY_PARAMS.get(strategy, {})
    video = VideoAsset(name=request.modality_path, num_frames=num_frames, strategy=strategy, strategy_params=strategy_params, smart_resize=smart_resize).np_ndarrays
    frames = []
    for frame in video:
        image = load_image_from_array(frame)
        frames.append(image)

    if model.alias.startswith("deepseek-vl2"):
        frame_placeholders = "".join(
            f"image_{i}:<image>\n" for i, _ in enumerate(frames, start=1)
        )
        return {
            "prompt": f"<|User|>: {frame_placeholders}{request.input}\n\n<|Assistant|>:",
            "multi_modal_data": {"image": frames}
        }

    frame_placeholders = [{"type": "image"}]  * len(frames)
    prompt = [
        {
            "role": "user",
            "content": [
                *frame_placeholders,
                {"type": "text", "text": request.input}
            ]
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )

    final_prompt = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": frames}
    }
    return final_prompt

def prepare_video_prompt(request: Request, model: Model, num_frames: int = None, strategy: str = "uniform", smart_resize: bool = False) -> Dict:
    processor = AutoProcessor.from_pretrained(model.path, use_fast=True)
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": request.input}
            ]
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )

    num_available_frames = request.modality_size["frame_count"]
    if num_available_frames == 0:
        return {}

    if not num_frames:
        num_frames = duration_to_frames(request.modality_size["duration"])
    num_frames = num_available_frames if num_available_frames < num_frames else num_frames
    
    strategy_params = ALL_STRATEGY_PARAMS.get(strategy, {})
    video = VideoAsset(name=request.modality_path, num_frames=num_frames, strategy=strategy, strategy_params=strategy_params, smart_resize=smart_resize).np_ndarrays

    final_prompt = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"video": video}
    }
    return final_prompt

def prepare_audio_prompt(request: Request, model: Model) -> Dict:
    processor = AutoProcessor.from_pretrained(model.path, use_fast=True)
    audio_in_prompt = "".join([
        f"Audio 1: "
        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n"
    ])
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": audio_in_prompt + request.input}
            ]
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )

    audio = AudioAsset(request.modality_path).audio_and_sample_rate

    final_prompt = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"audio": audio}
    }
    return final_prompt

def get_modality_token_index(request: Request, model: Model, multi_image: bool = False) -> int:
    image_codecs = {"JPEG"}
    video_codecs = {"h264", "vp6f", "vp9", "mp4"}
    audio_codecs = {"pcm_s16le"}
    
    if request.modality_path is None:
        return -1 

    if request.modality_size["codec"] in image_codecs:
        return model.image_token_index or -1
    elif request.modality_size["codec"] in video_codecs:
        if multi_image:
            return model.image_token_index or -1
        return model.video_token_index or -1
    elif request.modality_size["codec"] in audio_codecs:
        return model.audio_token_index or -1
    else:
        return -1
    
def is_video(request: Request) -> bool:
    return request.modality_size["codec"] in {"h264", "vp6f", "vp9", "mp4"}

def prepare_final_prompt(request: Request, model: Model, multi_image: bool = False,
                         num_frames: int = None, strategy: str = "uniform",
                         compression_ratio: float = None, smart_resize: bool = False) -> Dict:
    modality_token_index = get_modality_token_index(request, model, multi_image)

    if multi_image and modality_token_index != -1:
        if is_video(request):
            final_prompt = prepare_multi_image_prompt(request, model, num_frames, strategy, smart_resize)
        else:
            final_prompt = prepare_image_prompt(request, model, compression_ratio)
        

    elif modality_token_index == model.image_token_index:
        final_prompt = prepare_image_prompt(request, model, compression_ratio)

    elif modality_token_index == model.video_token_index:
        final_prompt = prepare_video_prompt(request, model, num_frames, strategy, smart_resize)

    elif modality_token_index == model.audio_token_index:
        final_prompt = prepare_audio_prompt(request, model)

    else:
        final_prompt = prepare_text_prompt(request, model)

    return final_prompt