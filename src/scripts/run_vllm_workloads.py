import json
import os
import time

from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.config.approaches import get_approach_by_name
from src.config.models import get_model_by_name
from src.config.workloads import get_workload_by_name
from src.constants import EXPERIMENTS_LOG, EXPERIMENTS_OUTPUTS_DIR
from src.utils import load_output

if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

    GPU_UTIL = 0.95
    SWAP_SPACE = 0
    approach = get_approach_by_name("Vanilla vLLM")

    workload_names = [
        "Text Conversations with Poisson 0.5",
        "Mixed Modalities with Poisson 0.5 15%"
    ]

    model_names = ["Mistral-7b"] * len(workload_names)

    # Read static results and get modality tokens and output tokens
    text_iso = load_output(os.path.join(EXPERIMENTS_OUTPUTS_DIR, "text-static-text-mistral-iso-20250107-125738.jsonl"))
    image_iso = load_output(os.path.join(EXPERIMENTS_OUTPUTS_DIR, "image-static-image-mistral-iso-20250107-125738.jsonl"))
    video_iso = load_output(os.path.join(EXPERIMENTS_OUTPUTS_DIR, "video-static-video-mistral-iso-20250107-222022.jsonl"))
    audio_iso = load_output(os.path.join(EXPERIMENTS_OUTPUTS_DIR, "audio-static-audio-qwen-iso-20250107-210445.jsonl"))

    text_id_pool = { o["id"] for o in text_iso }
    image_id_pool = { o["id"] for o in image_iso }
    video_id_pool = { o["id"] for o in video_iso }
    audio_id_pool = { o["id"] for o in audio_iso }

    iso_outputs = {o["id"]: o for o in text_iso + image_iso + video_iso + audio_iso}
    
    for workload_name, model_name in zip(workload_names, model_names):
        try:
            model = get_model_by_name(model_name)

            llm = LLM(
                model=model.path,
                gpu_memory_utilization=GPU_UTIL,
                swap_space=SWAP_SPACE,
                scheduling_policy=approach.scheduling_policy,
                enable_custom_scheduler=approach.enable_custom_scheduler,
                enable_chunked_prefill=approach.enable_chunked_prefill
            )

            workload = get_workload_by_name(workload_name)
            workload.load()
            requests = workload.requests
            timestamps = workload.timestamps
            
            pbar = tqdm(total=len(requests), desc='Finished requests')

            outputs = []
            request_cnt_to_id = {}
            start_time = time.time()
            while ((requests and timestamps) or llm.llm_engine.has_unfinished_requests()):
                now = time.time()

                while requests:
                    if timestamps[0] <= now - start_time:
                        request = requests.pop(0)
                        request_time = timestamps.pop(0)

                        request_id = str(next(llm.request_counter))
                        request_cnt_to_id[request_id] = request.id

                        sampling_params = SamplingParams(
                            ignore_eos=True,
                            max_tokens=iso_outputs[request.id]["decode_tokens_cnt"]
                        )

                        # Create a replacement prompt of equal length
                        replacement_prompt = " ".join("A" * (iso_outputs[request.id]["prompt_tokens_cnt"]-1))

                        # Add the overhead based on the modality
                        overhead = 0.0
                        if request.id not in text_id_pool:
                            overhead = iso_outputs[request.id]["processor_time"] + iso_outputs[request.id]["encoder_time"]

                        llm.llm_engine.add_request(
                            request_id=request_id,
                            prompt=replacement_prompt,
                            params=sampling_params,
                            arrival_time=start_time + request_time,
                            overhead=overhead
                        )
                    else:
                        break
                
                step_outputs = llm.llm_engine.step()

                now = time.time()
                for req_output in step_outputs:
                    if req_output.finished:
                        original_request_id = request_cnt_to_id[req_output.request_id]
                        # Set the modality_tokens_cnt based on the request.id (and the modality of the request)
                        if original_request_id in text_id_pool:
                            modality_tokens_cnt = 0
                        else:
                            modality_tokens_cnt = iso_outputs[original_request_id]["modality_tokens_cnt"]
                        
                        outputs.append({
                            "id": original_request_id,
                            "prompt_tokens_cnt": len(req_output.prompt_token_ids),
                            "modality_tokens_cnt": modality_tokens_cnt,
                            "decode_tokens_cnt": len(req_output.outputs[0].token_ids),
                            "processor_time": req_output.metrics.processor_time,
                            "encoder_time": req_output.metrics.encoder_time if req_output.metrics.encoder_time else 0,
                            "ttft": req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time,
                            "tbt": 0 if len(req_output.outputs[0].token_ids) <= 1 else (req_output.metrics.finished_time - req_output.metrics.first_token_time) / (len(req_output.outputs[0].token_ids)-1),
                            "e2e": req_output.metrics.finished_time - req_output.metrics.first_scheduled_time,
                            
                            "arrival_time": req_output.metrics.arrival_time,
                            "last_token_time": req_output.metrics.last_token_time,
                            "first_scheduled_time": req_output.metrics.first_scheduled_time,
                            "first_token_time": req_output.metrics.first_token_time,
                            "time_in_queue": req_output.metrics.time_in_queue,
                            "finished_time": req_output.metrics.finished_time,
                            "scheduler_time": req_output.metrics.scheduler_time,
                            "model_forward_time": req_output.metrics.model_forward_time,
                            "model_execute_time": req_output.metrics.model_execute_time
                        })
                        pbar.update(1)
        finally:
            pbar.close()
            elapsed_time = now - start_time

            total_num_tokens = 0
            for output in outputs:
                prompt_tokens_cnt = output["prompt_tokens_cnt"]
                decode_tokens_cnt = output["decode_tokens_cnt"]

                total_num_tokens += prompt_tokens_cnt + decode_tokens_cnt
    
            system_output = {
                "id": f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}",
                "elapsed_time": elapsed_time,
                "num_requests": len(outputs),
                "total_num_tokens": total_num_tokens,
                "requests_per_second": len(outputs) / elapsed_time,
                "tokens_per_second": total_num_tokens / elapsed_time
            }

            file = f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}.jsonl"
            path = os.path.join(EXPERIMENTS_OUTPUTS_DIR, file)
            with open(path, "w", encoding="utf-8") as file:
                for output in outputs:
                    file.write(json.dumps(output) + "\n")

            with open(EXPERIMENTS_LOG, "a", encoding="utf-8") as file:
                file.write(json.dumps(system_output) + "\n")
            
            llm = None