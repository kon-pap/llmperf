import gc
import json
import torch
import os
import numpy as np

from tqdm import tqdm
from vllm import LLM, SamplingParams

from constants import MODELS, ARTIFACTS_DIR

if __name__ == '__main__':
    stats = {}
    for model, meta in MODELS.items():
        llm = LLM(
            model=meta["model_path"],
            gpu_memory_utilization=0.95,
            swap_space=0
        )


        # Start - Warm up
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=16,
        )
        
        dummy_prompt_token_ids = np.random.randint(10000, size=(1,128))
        dummy_prompts = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]

        llm.generate(dummy_prompts,
                             sampling_params=sampling_params,
                             use_tqdm=False)
        
        # End - Warm up

        prompt_lengths = [1164, 1262, 1272, 1360, 1368, 1376, 1392, 1416, 1458, 1464, 1476, 1488, 1512, 1556, 1560, 1608, 1654, 1704, 1726, 1752, 1776, 1826, 1848, 1850, 1872, 1876, 1896, 1948, 1968, 2046, 2064, 2088, 2136, 2144, 2160, 2242, 2256, 2280, 2340, 2352, 2438, 2448, 2536, 2544, 2634, 2640, 2732, 2736, 2830, 2832, 2928]

        stats["Only Text"] = []
        for prompt_length in tqdm(prompt_lengths): 
            sampling_params = SamplingParams(
                ignore_eos=True,
                max_tokens=1
            )

            final_prompt = {
                "prompt": ' '.join(['A'] * (prompt_length-1)),
            }
            
            # Single prompt inference
            req_output = llm.generate(
                final_prompt,
                sampling_params,
                use_tqdm=False
            )[0]
            
            assert len(req_output.prompt_token_ids) == prompt_length

            stats["Only Text"].append({
                "prompt_tokens_cnt": len(req_output.prompt_token_ids),
                "image_tokens_cnt": 0,
                "image_size_pixel": 0,
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
        
        file = f"{meta['alias']}_only_text_stats.json" 
        path = os.path.join(ARTIFACTS_DIR, file)
        with open(path, "w") as f:
            json.dump(stats, f)

        # Kill LLM Engine
        del llm.llm_engine.model_executor
        del llm
        gc.collect()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()