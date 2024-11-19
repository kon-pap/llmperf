import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, ".."))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

MODELS_DIR = "/srv/muse-lab/models"
MODELS = {
    "LLaVA 1.6 (Mistral-7b)": {
        "model_path": "llava-hf/llava-v1.6-mistral-7b-hf", # os.path.join(MODELS_DIR, "llava-v1.6-mistral-7b-hf"),
        "max_model_len": 32768,
        "alias": "mistral"
    },
    "LLaVA 1.6 (Vicuna-7b)": {
        "model_path": "llava-hf/llava-v1.6-vicuna-7b-hf", # os.path.join(MODELS_DIR, "llava-v1.6-vicuna-7b-hf"),
        "max_model_len": 4096,
        "alias": "vicuna"
    },
}

DATASETS_DIR = "/srv/muse-lab/datasets"
DATASETS = {
    "Visual Q&A": {
        "path": os.path.join(DATASETS_DIR, "A-OKVQA"),
        "file": "aokvqa.jsonl",
        "alias": "vqa",
        "color": "#F8DE4B",
        # Question: <question> Answer: (https://arxiv.org/pdf/2206.01718)

        # <question> Answer the question using a single word or phrase. (https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)

        # <question> (https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
        # A. <option_1>
        # B. <option_2>
        # C. <option_3>
        # D. <option_4>
        # Answer with the option's letter from the given choices directly.

        # Based on the above image, please answer the question. <question> Please provide an accurate answer within one word. The answer is: (Custom https://arxiv.org/pdf/2401.10208)
    },
    "Image Captioning": {
        "path": os.path.join(DATASETS_DIR, "NoCaps"),
        "file": "nocaps.jsonl",
        "alias": "caps",
        "color": "#577FBC",
        # Describe the image concisely. (https://arxiv.org/pdf/2304.08485)
        # Provide a one-sentence caption for the provided image. (https://arxiv.org/pdf/2401.10208)
    },
    "Detailed Description": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        "file": "detail.jsonl",
        "alias": "detail",
        "color": "#E16F65",
        # Describe the following image in detail (https://arxiv.org/pdf/2304.08485)
        # Please describe the image in detail. (https://arxiv.org/pdf/2401.10208)
    },
    "Complex Reasoning": {
        "path": os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        "file": "complex_reasoning.jsonl",
        "alias": "complex",
        "color": "#57B593",
        # Question: <question> Answer: (https://arxiv.org/pdf/2206.01718)
        # <no post processing> (https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
    },
}