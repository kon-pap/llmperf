import io
import numpy as np

from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoTokenizer
from typing import Dict

from vllm.assets.audio import AudioAsset
from vllm.assets.video import VideoAsset

from llmperf.config.models import Model
from llmperf.config.workloads import Request

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

def prepare_image_prompt(request: Request, model: Model) -> Dict:
    processor = AutoProcessor.from_pretrained(model.path, use_fast=True)

    if processor.chat_template is None:
        image = load_image(request.modality_path)
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
    
    image = load_image(request.modality_path)

    if image.mode != "RGB" and model.alias.startswith("gemma-3"):
        image = image.convert("RGB")

    final_prompt = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image}
    }
    return final_prompt

def prepare_multi_image_prompt(request: Request, model: Model) -> Dict:
    processor = AutoProcessor.from_pretrained(model.path, use_fast=True)

    num_available_frames = request.modality_size["frame_count"]
    if num_available_frames == 0:
        return {}

    num_frames = duration_to_frames(request.modality_size["duration"])
    num_frames = num_available_frames if num_available_frames < num_frames else num_frames

    video = VideoAsset(name=request.modality_path, num_frames=num_frames).np_ndarrays
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

    frame_placeholders = [{"type": "image"}]  * num_frames
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

def prepare_video_prompt(request: Request, model: Model) -> Dict:
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

    num_frames = duration_to_frames(request.modality_size["duration"])
    num_frames = num_available_frames if num_available_frames < num_frames else num_frames
    
    video = VideoAsset(name=request.modality_path, num_frames=num_frames).np_ndarrays

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
    video_codecs = {"h264", "vp6f", "vp9"}
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
    
def prepare_final_prompt(request: Request, model: Model, multi_image: bool = False) -> Dict:
    modality_token_index = get_modality_token_index(request, model)

    if multi_image and modality_token_index == -1:
        final_prompt = prepare_multi_image_prompt(request, model)

    elif modality_token_index == model.image_token_index:
        final_prompt = prepare_image_prompt(request, model)

    elif modality_token_index == model.video_token_index:
        final_prompt = prepare_video_prompt(request, model)

    elif modality_token_index == model.audio_token_index:
        final_prompt = prepare_audio_prompt(request, model)

    else:
        final_prompt = prepare_text_prompt(request, model)

    return final_prompt