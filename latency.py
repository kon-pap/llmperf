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
    
    for model, model_meta in MODELS.items():
        llm = LLM(
            model=model_meta["model_path"],
            gpu_memory_utilization=0.95,
            swap_space=0
        )

        tokenizer = AutoTokenizer.from_pretrained(model_meta["model_path"])

        stats = {}
        for dataset, ds_meta in DATASETS.items():
            # Load dataset
            ds_path = os.path.join(ds_meta["path"], ds_meta["file"])
            data = load_dataset(ds_path)

            # Get prompts
            if ds_meta["alias"] in ["detail", "complex"]:
                prompts = get_prompts(ds_meta["alias"])(data, use_case=dataset)
            else:
                prompts = get_prompts(ds_meta["alias"])(data)
            
            # Get image paths
            dir = os.path.join(ds_meta["path"], "images")
            image_paths = get_image_paths(ds_meta["alias"])(data, dir)

            # Get output lengths
            lengths = get_lengths(ds_meta["alias"])(tokenizer, data)
            input_lengths, output_lengths = map(list, zip(*lengths))
            
            stats[dataset] = {}
            for image_flag, prompt_flag in [(True, True), (True, False), (False, True)]:
                if image_flag and prompt_flag:
                    conf = "With Image"
                elif not image_flag and prompt_flag:
                    conf = "Without Image"
                else:
                    conf = "Only Image"
                
                stats[dataset][conf] = []

                print(f"Executing: {model} - {dataset} - {conf}")
                for prompt, image_path, output_length, input_length in tqdm(zip(prompts[:SAMPLES], image_paths[:SAMPLES], output_lengths[:SAMPLES], input_lengths[:SAMPLES]), total=len(prompts[:SAMPLES])):
                    formatted_prompt = generate_prompt(model_meta["model_path"], prompt, image_flag=image_flag, prompt_flag=prompt_flag)

                    # Load the image using PIL.Image
                    image = load_image(image_path)
                    
                    sampling_params = SamplingParams(
                        ignore_eos=True,
                        max_tokens=int(output_length)
                    )

                    if image_flag:
                        final_prompt = {
                            "prompt": formatted_prompt,
                            "multi_modal_data": {"image": image},
                        }
                    else:
                        final_prompt = {
                            "prompt": formatted_prompt,
                        }
                    # Single prompt inference
                    req_output = llm.generate(
                        final_prompt,
                        sampling_params,
                        use_tqdm=False
                    )[0]
                    
                    # assert len(req_output.outputs[0].token_ids) == int(output_length)
                    
                    stats[dataset][conf].append({
                        "raw_prompt_tokens_cnt": input_length,
                        "prompt_tokens_cnt": len(req_output.prompt_token_ids),
                        "image_tokens_cnt": req_output.prompt_token_ids.count(32000),
                        "image_size_pixel": image.size,
                        "decode_tokens_cnt": len(req_output.outputs[0].token_ids),
                        "processor_time": req_output.metrics.processor_time,
                        "encoder_time": req_output.metrics.encoder_time,
                        "text_time": req_output.metrics.text_time,
                        "visual_time": req_output.metrics.visual_time,
                        "merge_time": req_output.metrics.merge_time,
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
        torch.cuda.empty_cache()