import json
import os
import random
import numpy as np

from PIL import Image, ImageFile
from typing import List, Tuple
from transformers import LlavaNextProcessor, AutoTokenizer
from vllm import SamplingParams


def load_dataset(dataset_path):
    with open(dataset_path) as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        return data
    
def get_vqa(tokenizer: AutoTokenizer, data: List[dict], long: bool = False) -> List[Tuple[int, int]]:
    lengths = []
    for record in data:
        input_tokens = tokenizer.encode(record["question"])[1:]

        outputs = record["rationales"] if long else record["direct_answers"]
        output_tokens = [tokenizer.encode(output)[1:] for output in outputs]

        lengths.append((
            len(input_tokens),
            sum([len(single_output_tokens) for single_output_tokens in output_tokens]) / len(output_tokens)
        ))
        
    return lengths

def get_caps(tokenizer: AutoTokenizer, data: List[dict]) -> List[Tuple[int, int]]:
    lengths = []
    for record in data:
        input_tokens = [] # Image Captioning use case doesn't have any context

        output_tokens = [tokenizer.encode(output)[1:] for output in record["annotations_captions"]]

        lengths.append((
            len(input_tokens),
            sum([len(single_output_tokens) for single_output_tokens in output_tokens]) / len(output_tokens)
        ))
    
    return lengths

def get_llava(tokenizer: AutoTokenizer, data: List[dict]) -> List[Tuple[int, int]]:
    lengths = []
    for record in data:
        if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
            continue
        
        input_tokens = tokenizer.encode(record["conversations"][0]["value"].replace("<image>\n", ""))[1:]

        output_tokens = tokenizer.encode(record["conversations"][1]["value"])[1:]

        lengths.append((len(input_tokens), len(output_tokens)))
        
    return lengths

def get_vqa_prompts(data: List[dict]) -> List[str]:
    prompts = []
    for record in data:
        input_seq = record["question"] + "\nAnswer the question using a single word or phrase."
        prompts.append(input_seq)
        
    return prompts

def get_caps_prompts(data: List[dict]) -> List[str]:
    return ["Describe the image concisely."] * len(data)

def get_llava_prompts(data: List[dict], use_case: str = "Detailed Descirption") -> List[str]:
    prompts = []
    for record in data:
        if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
            continue
        
        input_seq = record["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")

        if use_case == "Detailed Descirption":
            input_seq = input_seq + "\nPlease describe the image in detail."
        
        prompts.append(input_seq)
        
    return prompts

def get_vqa_image_paths(data: List[dict], dir: str) -> List[str]:
    image_paths = []
    for record in data:
        image_path = os.path.join(dir, f"{record['question_id']}.jpg")
        image_paths.append(image_path)
        
    return image_paths

def get_caps_image_paths(data: List[dict], dir: str) -> List[str]:
    image_paths = []
    for record in data:
        image_path = os.path.join(dir, record["image_file_name"])
        image_paths.append(image_path)
        
    return image_paths

def get_llava_image_paths(data: List[dict], dir: str) -> List[str]:
    image_paths = []
    for record in data:
        if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
            continue
        
        image_path = os.path.join(dir, record["image"])
        image_paths.append(image_path)
        
    return image_paths

def get_lengths(dataset_alias):
    functions = {
        "vqa": get_vqa,
        "caps": get_caps,
        "detail": get_llava,
        "complex": get_llava
    }

    # Return the selected function or a default one if the key doesn't exist
    return functions.get(dataset_alias, lambda: None)

def get_prompts(dataset_alias):
    functions = {
        "vqa": get_vqa_prompts,
        "caps": get_caps_prompts,
        "detail": get_llava_prompts,
        "complex": get_llava_prompts
    }

    # Return the selected function or a default one if the key doesn't exist
    return functions.get(dataset_alias, lambda: None)

def get_image_paths(dataset_alias):
    functions = {
        "vqa": get_vqa_image_paths,
        "caps": get_caps_image_paths,
        "detail": get_llava_image_paths,
        "complex": get_llava_image_paths
    }

    # Return the selected function or a default one if the key doesn't exist
    return functions.get(dataset_alias, lambda: None)

def generate_prompt(model: str, prompt: str, image_flag: bool = True, prompt_flag: bool = True) -> str:
    processor = LlavaNextProcessor.from_pretrained(model)

    conversation = [
        {
            "role": "user",
            "content": [],
        }
    ]
    if image_flag:
        conversation[0]["content"].append({"type": "image"})

    if prompt_flag:
        conversation[0]["content"].append({"type": "text", "text": prompt})

    fromatted_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )

    return fromatted_prompt

def load_image(path: str) -> ImageFile.ImageFile:
    return Image.open(path)

def sample_requests(
    dataset: str,
    ds_meta: dict,
    num_requests: int,
    model: str,
    tokenizer: AutoTokenizer,
    image_flag: bool,
) -> List[Tuple[dict, int]]:
    # Load dataset
    ds_path = os.path.join(ds_meta["path"], ds_meta["file"])
    data = load_dataset(ds_path)

    # Get prompts
    if ds_meta["alias"] in ["detail", "complex"]:
        prompts = get_prompts(ds_meta["alias"])(data, use_case=dataset, num_of_records=num_requests)
    else:
        prompts = get_prompts(ds_meta["alias"])(data, num_of_records=num_requests)
    
    # Get image paths
    dir = os.path.join(ds_meta["path"], "images")
    image_paths = get_image_paths(ds_meta["alias"])(data, dir, num_of_records=num_requests)

    # Get output lengths
    lengths = get_lengths(ds_meta["alias"])(tokenizer, data, num_of_records=num_requests)
    _, output_lengths = map(list, zip(*lengths))

    requests = []
    for prompt, image_path, output_length in zip(prompts, image_paths, output_lengths):
        if len(requests) == num_requests:
            break

        formatted_prompt = generate_prompt(model, prompt, image_flag=image_flag)

        # Load the image using PIL.Image
        image = load_image(image_path)

        if image_flag:
            final_prompt = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image},
            }
        else:
            final_prompt = {
                "prompt": formatted_prompt,
            }

        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=int(output_length),
        )
        
        requests.append((final_prompt, sampling_params))
    
    return requests

def sample_requests_trace(
    dataset: str,
    ds_meta: dict,
    request_rate: float,
    duration: int,
    model: str,
    tokenizer: AutoTokenizer,
    image_flag: bool,
    seed: int = 0,
    time_quantum: int = 10
) -> List[Tuple[dict, int]]:
    random.seed(seed)
    np.random.seed(seed)

    # Generate timestamps for requests using Poisson distribution.
    lam = request_rate * (time_quantum / 1000)
    quantums_per_sec = 1000 / time_quantum
    arrival_times = np.random.poisson(
        lam=lam, size=int(duration * quantums_per_sec))
    timestamps = []
    for i, n in enumerate(arrival_times):
        timestamps += [i * (time_quantum / 1000)] * n

    num_requests = len(timestamps)

    requests = sample_requests(
        dataset=dataset,
        ds_meta=ds_meta,
        num_requests=num_requests,
        model=model,
        tokenizer=tokenizer,
        image_flag=image_flag
    )

    random.shuffle(requests)

    trace = []
    for timestamp, request in zip(timestamps, requests):
        trace.append((timestamp, request[0], request[1]))

    return trace

def find_mean_coord(x, y):
    x_mean = np.mean(x)
    
    mean_idx = 0
    for i, x_val in enumerate(x):
        if x_val >= x_mean:
            mean_idx = i
            break

    return x_mean, y[mean_idx]

def get_cdf(data):
    N = len(data)

    x = np.sort(data)
    y = np.arange(N) / float(N)

    return x, y
