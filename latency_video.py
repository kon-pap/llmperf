import gc
import json
import torch
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import MODELS, DATASETS, ARTIFACTS_DIR
from utils import load_dataset, get_prompts, get_video_paths, get_lengths, generate_video_prompt
from vllm.assets.video import VideoAsset

if __name__ == '__main__':
    SAMPLES = 1000
    FRAMES = 32
    model = "LLaVA-Next-Video (Mistral-7b)"
    model_meta = MODELS[model]

    llm = LLM(
        model=model_meta["model_path"],
        gpu_memory_utilization=0.95,
        swap_space=0
    )

    tokenizer = AutoTokenizer.from_pretrained(model_meta["model_path"])

    stats = {}
    dataset = "Video Description"
    ds_meta = DATASETS[dataset]

    # Load dataset
    ds_path = os.path.join(ds_meta["path"], ds_meta["file"])
    data = load_dataset(ds_path)

    # Get prompts
    prompts = get_prompts(ds_meta["alias"])(data, use_case=dataset)
    
    # Get video paths
    dir = os.path.join(ds_meta["path"], "videos")
    video_paths = get_video_paths(ds_meta["alias"])(data, dir)

    # Get output lengths
    lengths = get_lengths(ds_meta["alias"])(tokenizer, data)
    input_lengths, output_lengths = map(list, zip(*lengths))
    
    stats[dataset] = {}
    for video_flag, prompt_flag in [(True, True), (False, True)]:
        if video_flag and prompt_flag:
            conf = f"With Video ({FRAMES} frames)"
        else:
            conf = "Without Video"
        
        stats[dataset][conf] = []

        print(f"Executing: {model} - {dataset} - {conf}")
        for prompt, video_path, output_length, input_length in tqdm(zip(prompts[:SAMPLES], video_paths[:SAMPLES], output_lengths[:SAMPLES], input_lengths[:SAMPLES]), total=len(prompts[:SAMPLES])):
            formatted_prompt = generate_video_prompt(model_meta["model_path"], prompt, video_flag=video_flag, prompt_flag=prompt_flag)

            # Load the video
            video = VideoAsset(name=video_path,
                           num_frames=FRAMES).np_ndarrays
            
            sampling_params = SamplingParams(
                ignore_eos=True,
                max_tokens=int(output_length)
            )

            if video_flag:
                final_prompt = {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"video": video},
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
            
            stats[dataset][conf].append({
                "raw_prompt_tokens_cnt": input_length,
                "prompt_tokens_cnt": len(req_output.prompt_token_ids),
                "video_tokens_cnt": req_output.prompt_token_ids.count(model_meta["video_token_index"]),
                "video_size": video.shape,
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