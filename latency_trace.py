import gc
import json
import torch
import time
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import ARTIFACTS_DIR, MODELS, DATASETS
from utils import sample_requests_trace

if __name__ == '__main__':
    for model, model_meta in MODELS.items():
        llm = LLM(
            model=model_meta["model_path"],
            gpu_memory_utilization=0.95,
            swap_space=0
        )

        tokenizer = AutoTokenizer.from_pretrained(model_meta["model_path"])

        ### Start - Warm-Up
        num_warmup_requests = 8
        warmup_input_len = 8
        warmup_output_len = 32
        warmup_sampling_params = SamplingParams(
            n=1,
            best_of=1,
            temperature=1.0,
            top_p=0.99,
            ignore_eos=True,
            max_tokens=int(warmup_output_len)
        )

        llm.generate(
            sampling_params=warmup_sampling_params,
            prompt_token_ids=[[0] * warmup_input_len for _ in range(num_warmup_requests)]
        )
        ### End - Warm-Up

        stats = {}
        for dataset, ds_meta in DATASETS.items():
            stats[dataset] = {}
            for image_flag in [True, False]:
                duration=100
                if image_flag:
                    request_rate = 5
                    conf = "With Image"
                elif not image_flag:
                    request_rate = 10
                    conf = "Without Image"
                
                stats[dataset][conf] = {}
                
                print(f"Experiment: {model} - {dataset} - {conf} - {request_rate}")
                print("Creating Requests...")
                requests = sample_requests_trace(
                    dataset=dataset,
                    ds_meta=ds_meta,
                    request_rate=request_rate,
                    duration=duration,
                    model=model_meta["model_path"],
                    tokenizer=tokenizer,
                    image_flag=image_flag
                )

                print("Executing Requests...")
                pbar = tqdm(total=len(requests), desc='Finished requests')

                outputs = []
                start_time = time.monotonic()
                while (requests or llm.llm_engine.has_unfinished_requests()):
                    now = time.monotonic()

                    while requests:
                        if requests[0][0] <= now - start_time:
                            request_time, prompt, sampling_params = requests.pop(0)
                            request_id = str(next(llm.request_counter))
                            llm.llm_engine.add_request(
                                request_id=request_id,
                                prompt=prompt,
                                params=sampling_params,
                                arrival_time=start_time + request_time)
                        else:
                            break
                    
                    step_outputs = llm.llm_engine.step()

                    now = time.monotonic()
                    for req_output in step_outputs:
                        if req_output.finished:
                            outputs.append({
                                "prompt_tokens_cnt": len(req_output.prompt_token_ids),
                                "image_tokens_cnt": req_output.prompt_token_ids.count(32000),
                                "decode_tokens_cnt": len(req_output.outputs[0].token_ids),
                                "processor_time": req_output.metrics.processor_time,
                                "encoder_time": req_output.metrics.encoder_time,
                                "ttft": req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time,
                                "tbt": 0 if len(req_output.outputs[0].token_ids) <= 1 else (req_output.metrics.finished_time - req_output.metrics.first_token_time) / (len(req_output.outputs[0].token_ids)-1),
                                "e2e": req_output.metrics.finished_time - req_output.metrics.first_scheduled_time,
                                "arrival_time": req_output.metrics.arrival_time,
                                "last_token_time": req_output.metrics.last_token_time,
                                "time_in_queue": req_output.metrics.time_in_queue,
                                "scheduler_time": req_output.metrics.scheduler_time
                            })
                            pbar.update(1)

                pbar.close()
                elapsed_time = now - start_time

                total_num_tokens = 0
                for output in outputs:
                    prompt_tokens_cnt = output["prompt_tokens_cnt"]
                    decode_tokens_cnt = output["decode_tokens_cnt"]

                    total_num_tokens += prompt_tokens_cnt + decode_tokens_cnt
        
                stats[dataset][conf] = {
                    "outputs": outputs,
                    "elapsed_time": elapsed_time,
                    "num_requests": len(outputs),
                    "total_num_tokens": total_num_tokens,
                    "requests_per_second": len(outputs) / elapsed_time,
                    "tokens_per_second": total_num_tokens / elapsed_time,
                    "request_rate": request_rate,
                    "duration": duration
                }
            
        file = f"{model_meta['alias']}_latency_stats.json" 
        path = os.path.join(ARTIFACTS_DIR, file)
        with open(path, "w") as f:
            json.dump(stats, f)

        # Kill LLM Engine
        del llm.llm_engine.model_executor
        del llm
        gc.collect()
        torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()