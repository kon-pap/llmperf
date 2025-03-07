import ffmpeg
import os

from PIL import Image
from typing import Dict, LiteralString, Tuple, Union

def get_image_size(dir: Union[str,LiteralString], record: Dict) -> Dict:
    path = os.path.join(dir, "images", record["image"])

    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "color_mode": img.mode,
            "file_size": os.path.getsize(path),
            "codec": img.format
        }

def get_video_size(dir: Union[str,LiteralString], record: Dict) -> Dict:
    path = os.path.join(dir, "videos", record["video"])
    
    probe = ffmpeg.probe(path)
    stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    fmt = probe["format"]

    return {
        "duration": int(float(fmt["duration"])),
        "frame_count": int(stream.get("nb_frames", 0)),
        "bit_rate": int(fmt["bit_rate"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "color_mode": stream["pix_fmt"],
        "file_size": float(fmt["size"]),
        "codec": stream["codec_name"]
    }

def get_audio_size(dir: Union[str,LiteralString], record: Dict) -> Dict:
    path = os.path.join(dir, "audios", record["audio_id"])
    
    probe = ffmpeg.probe(path)
    stream, fmt = probe["streams"][0], probe["format"]

    return {
        "duration": int(float(fmt["duration"])),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "bit_rate": int(fmt["bit_rate"]),
        "file_size": float(fmt["size"]),
        "codec": stream["codec_name"]
    }