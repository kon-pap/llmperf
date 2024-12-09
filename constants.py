import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, ".."))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

MODELS_DIR = "/srv/muse-lab/models"
MODELS = {
    "Mistral-7b": {
        "model_path": os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.2"), # "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 32768,
        "alias": "text_mistral",
    },
    "LLaVA 1.6 (Mistral-7b)": {
        "model_path": "llava-hf/llava-v1.6-mistral-7b-hf",
        "max_model_len": 32768,
        "alias": "image_mistral",
        "image_token_index": 32000
    },
    "LLaVA 1.6 (Vicuna-7b)": {
        "model_path": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "max_model_len": 4096,
        "alias": "image_vicuna",
        "image_token_index": 32000
    },
    "LLaVA-Next-Video (Mistral-7b)": {
        "model_path": "llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
        "max_model_len": 32768,
        "alias": "video_mistral",
        "image_token_index": 32001,
        "video_token_index": 32000,
    },
    "Qwen2-Audio-7b": {
        "model_path": "Qwen/Qwen2-Audio-7B-Instruct",
        "max_model_len": 8192,
        "alias": "audio_qwen",
        "audio_token_index": 151646
    }

}

DATASETS_DIR = "/srv/muse-lab/datasets"
DATASETS = {
    "Visual Q&A": {
        "path": os.path.join(DATASETS_DIR, "A-OKVQA"),
        "file": "aokvqa.jsonl",
        "alias": "vqa",
        "color": "#F8DE4B",
    },
    "Image Captioning": {
        "path": os.path.join(DATASETS_DIR, "NoCaps"),
        "file": "nocaps.jsonl",
        "alias": "imgcaps",
        "color": "#577FBC",
    },
    "Detailed Description": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        "file": "detail.jsonl",
        "alias": "detaildesc",
        "color": "#E16F65",
    },
    "Complex Reasoning": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        "file": "complex_reasoning.jsonl",
        "alias": "compreason",
        "color": "#57B593",
    },
    "Multiple-Choice Video Q&A": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Video"),
        "file": "multiple_choice.jsonl",
        "alias": "mcvidqa",
        "color": "#F8DE4B",
    },
    "Video Description": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Video"),
        "file": "description.jsonl",
        "alias": "viddesc",
        "color": "#E16F65",
    },
    "Open-Ended Video Q&A": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Video"),
        "file": "open_ended.jsonl",
        "alias": "oevidqa",
        "color": "#57B593",
    },
    "Multi-Image Q&A": {
        "path": os.path.join(DATASETS_DIR, "Mantis-Instruct"),
        "file": "multi_image_qa.jsonl",
        "alias": "miqa",
        "color": "#577FBC",
    },
    "Conversations": {
        "path": os.path.join(DATASETS_DIR, "ShareGPT"),
        "file": "sharegpt.jsonl",
        "alias": "conv",
        "color": "#577FBC",
    },
    "Audio Understanding": {
        "path": os.path.join(DATASETS_DIR, "Clotho"),
        "file": "understanding.jsonl",
        "alias": "audiound",
        "color": "#E16F65",
    },
}