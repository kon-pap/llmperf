import gc
import json
import torch
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import MODELS, DATASETS, ARTIFACTS_DIR
from utils import load_dataset, get_prompts, get_image_paths, get_lengths, generate_prompt, load_image

if __name__ == '__main__':
    SAMPLES = 1000
    model = "Mistral-7b"
    model_meta = MODELS[model]

    llm = LLM(
        model=model_meta["model_path"],
        gpu_memory_utilization=0.95,
        swap_space=0
    )

    tokenizer = AutoTokenizer.from_pretrained(model_meta["model_path"])

    stats = {}
    dataset = "Conversations"
    ds_meta = DATASETS[dataset]

    # Load dataset
    ds_path = os.path.join(ds_meta["path"], ds_meta["file"])
    data = load_dataset(ds_path)

    # Get prompts
    prompts = get_prompts(ds_meta["alias"])(data)

    # Get output lengths
    lengths = get_lengths(ds_meta["alias"])(tokenizer, data)
    input_lengths, output_lengths = map(list, zip(*lengths))
    

    stats[dataset] = []
    print(f"Executing: {model} - {dataset}")
    for prompt, output_length, input_length in tqdm(zip(prompts[:SAMPLES], output_lengths[:SAMPLES], input_lengths[:SAMPLES]), total=len(prompts[:SAMPLES])):
        formatted_prompt = prompt

        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=int(output_length)
        )

        final_prompt = {
            "prompt": formatted_prompt,
        }
            
        
        # Single prompt inference
        req_output = llm.generate(
            final_prompt,
            sampling_params,
            use_tqdm=False
        )[0]
        
        stats[dataset].append({
            "raw_prompt_tokens_cnt": input_length,
            "prompt_tokens_cnt": len(req_output.prompt_token_ids),
            "decode_tokens_cnt": len(req_output.outputs[0].token_ids),
            "processor_time": req_output.metrics.processor_time,
            "encoder_time": req_output.metrics.encoder_time,
            "ttft": req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time,
            "tbt": 0 if len(req_output.outputs[0].token_ids) <= 1 else (req_output.metrics.finished_time - req_output.metrics.first_token_time) / (len(req_output.outputs[0].token_ids)-1),
            "e2e": req_output.metrics.finished_time - req_output.metrics.first_scheduled_time,
        })

    file = f"{model_meta['alias']}_stats.json"
    path = os.path.join(ARTIFACTS_DIR, file)
    with open(path, "w") as f:
        json.dump(stats, f)

    # Kill LLM Engine
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()