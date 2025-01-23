import os

from dataclasses import dataclass
from typing import LiteralString, Optional, Union

from src.constants import MODELS_DIR

@dataclass
class Model:
    name: str
    path: Union[str,LiteralString]
    max_model_len: int
    alias: str
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    audio_token_index: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.name == other.name and self.alias == other.alias
        return False

MODELS = {
    Model(
        name="Mistral-7b",
        path="mistralai/Mistral-7B-Instruct-v0.2", # os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.2")
        max_model_len=32768,
        alias="text-mistral"
    ),
    Model(
        name="LLaVA 1.6 (Mistral-7b)",
        path="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=32768,
        alias="image-mistral",
        image_token_index=32000
    ),
    Model(
        name="LLaVA-Next-Video (Mistral-7b)",
        path="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
        max_model_len=32768,
        alias="video-mistral",
        image_token_index=32001,
        video_token_index=32000
    ),
    Model(
        name="Qwen2-Audio-7b",
        path="Qwen/Qwen2-Audio-7B-Instruct",
        max_model_len=8192,
        alias="audio-qwen",
        audio_token_index=151646
    )
}

def get_model_by_name(name: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "name", None) == name:
            return model
    return None

def get_model_by_name(alias: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "alias", None) == alias:
            return model
    return None