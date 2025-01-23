import ffmpeg
import os

from PIL import Image
from typing import Dict, LiteralString, Tuple, Union

def get_image_size(dir: Union[str,LiteralString], record: Dict) -> Tuple[int,int]:
    path = os.path.join(dir, "images", record["image"])

    with Image.open(path) as img:
        return img.size

def get_video_size(dir: Union[str,LiteralString], record: Dict) -> int:
    path = os.path.join(dir, "videos", record["video"])
    
    probe = ffmpeg.probe(path)
    return int(float(probe['format']['duration']))

def get_audio_size(dir: Union[str,LiteralString], record: Dict) -> int:
    path = os.path.join(dir, "audios", record["audio_id"])
    
    probe = ffmpeg.probe(path)
    return int(float(probe['format']['duration']))