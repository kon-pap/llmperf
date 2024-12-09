import gc
import json
import torch
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from transformers import AutoTokenizer

from constants import MODELS, DATASETS, ARTIFACTS_DIR
from utils import load_dataset, get_prompts, get_audio_paths, get_lengths, generate_audio_prompt, load_image

if __name__ == '__main__':
    SAMPLES = 1000
    model = "Qwen2-Audio-7b"
    model_meta = MODELS[model]

    llm = LLM(
        model=model_meta["model_path"],
        gpu_memory_utilization=0.95,
        swap_space=0
    )

    tokenizer = AutoTokenizer.from_pretrained(model_meta["model_path"])

    stats = {}
    dataset = "Audio Understanding"
    ds_meta = DATASETS[dataset]

    # Load dataset
    ds_path = os.path.join(ds_meta["path"], ds_meta["file"])
    data = load_dataset(ds_path)

    # Get prompts
    prompts = get_prompts(ds_meta["alias"])(data)

    # Get audio paths
    dir = os.path.join(ds_meta["path"], "audios")
    audio_paths = get_audio_paths(ds_meta["alias"])(data, dir)

    # Get output lengths
    lengths = get_lengths(ds_meta["alias"])(tokenizer, data)
    input_lengths, output_lengths = map(list, zip(*lengths))

    stats[dataset] = []

    print(f"Executing: {model} - {dataset}")
    i = 0
    for prompt, audio_path, output_length, input_length in tqdm(zip(prompts[:SAMPLES], audio_paths[:SAMPLES], output_lengths[:SAMPLES], input_lengths[:SAMPLES]), total=len(prompts[:SAMPLES])):
        formatted_prompt = generate_audio_prompt(model_meta["model_path"], prompt)
        
        # Load the audio
        audio = AudioAsset(audio_path).audio_and_sample_rate
        
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=int(output_length)
        )

        
        final_prompt = {
            "prompt": formatted_prompt,
            "multi_modal_data": {"audio": audio},
        }
        
        if i in [387, 388, 437, 438, 820, 821]:
            print(final_prompt)
            i = i + 1
            continue

        # Single prompt inference
        req_output = llm.generate(
            final_prompt,
            sampling_params,
            use_tqdm=False
        )[0]
        
        stats[dataset].append({
            "raw_prompt_tokens_cnt": input_length,
            "prompt_tokens_cnt": len(req_output.prompt_token_ids),
            "audio_tokens_cnt": req_output.prompt_token_ids.count(model_meta["audio_token_index"]),
            "audio_size": audio[0].shape,
            "decode_tokens_cnt": len(req_output.outputs[0].token_ids),
            "processor_time": req_output.metrics.processor_time,
            "encoder_time": req_output.metrics.encoder_time,
            "ttft": req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time,
            "tbt": 0 if len(req_output.outputs[0].token_ids) <= 1 else (req_output.metrics.finished_time - req_output.metrics.first_token_time) / (len(req_output.outputs[0].token_ids)-1),
            "e2e": req_output.metrics.finished_time - req_output.metrics.first_scheduled_time,
        })

        i = i + 1

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