import argparse
import time

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

from llmperf.config.approaches import get_approach_by_alias
from llmperf.config.models import get_model_by_alias
from llmperf.config.workloads import get_workload_by_alias
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import create_experiment_id, get_modality_token_index, prepare_final_prompt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse resource usage and experiment metadata."
    )
    
    parser.add_argument('--gpu-util', type=float, default=0.95,
                        help='GPU utilization percentage (default: 0.95)')
    parser.add_argument('--swap-space', type=int, default=0,
                        help='Swap space used in GB (default: 0)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model alias (e.g., llava-ov)')
    parser.add_argument('--workload', type=str, required=True,
                        help='Workload alias (e.g., video-static)')
    
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Model context length")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Maximum number of batched tokens per iteration")
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None,
                        help="Number of GPU blocks")

    parser.add_argument("--multi-image", action="store_true",
                        help="Use multiple images instead of video")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    approach = get_approach_by_alias("iso")
    model = get_model_by_alias(args.model)
    workload = get_workload_by_alias(args.workload)
    workload.load()

    try:

        limit_mm_per_prompt = None
        if args.multi_image and workload.alias == "video-static":
            limit_mm_per_prompt = {"image": 64}

        if args.multi_image == False and workload.alias == "video-static":
            limit_mm_per_prompt = {"video": 1}

        llm = LLM(
            model=model.path,
            gpu_memory_utilization=args.gpu_util,
            swap_space=args.swap_space,
            scheduling_policy=approach.scheduling_policy,
            disable_log_stats=False,
            max_model_len=args.max_model_len or model.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens or model.max_model_len,
            num_gpu_blocks_override=args.num_gpu_blocks_override or (model.max_model_len // 16),
            hf_token=True, # requires huggingface-cli login
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]} if model.alias.startswith("deepseek-vl2") else None,
            limit_mm_per_prompt=limit_mm_per_prompt,
            disable_mm_preprocessor_cache=True,
            max_num_seqs=1
        )

        requests = workload.requests
        tokenizer = AutoTokenizer.from_pretrained(model.path)

        outputs = []
        start_time = now = time.time()
        for request in tqdm(requests):
            output_length = len(tokenizer.encode(request.output))

            sampling_params = SamplingParams(
                ignore_eos=True,
                max_tokens=int(output_length)
            )

            modality_token_index = get_modality_token_index(request, model, multi_image=args.multi_image)
            final_prompt = prepare_final_prompt(request, model, multi_image=args.multi_image)

            req_outputs = llm.generate(
                prompts=[final_prompt],
                sampling_params=sampling_params,
                use_tqdm=False
            )

            now = time.time()
            if req_outputs:
                # Aborted request because prompt was longer than max model length
                processed_inputs = llm.llm_engine.processor.input_preprocessor.preprocess(
                    final_prompt,
                    lora_request=None,
                    prompt_adapter_request=None,
                    return_mm_hashes=False,
                )
                prompt_token_ids = processed_inputs["prompt_token_ids"]
                outputs.append(
                    RequestOutput(
                        request.id,
                        len(prompt_token_ids),
                        prompt_token_ids.count(modality_token_index),
                        aborted=True
                    )
                )
            else:
                req_output = req_outputs[0]
                outputs.append(
                    RequestOutput.from_vllm_output(
                        request.id, req_output, modality_token_index
                    )
                )

    finally:
        elapsed_time = now - start_time
        experiment_id = create_experiment_id(
            workload.alias,
            model.alias,
            approach.alias,
            args.gpu_util,
            args.swap_space,
            args.num_gpu_blocks_override,
            args.max_model_len,
            args.max_num_batched_tokens
        )

        if elapsed_time == 0:
            print(f"Failed {experiment_id}")
        else:
            experiment_output = ExperimentOutput(
                id=experiment_id,
                elapsed_time=elapsed_time,
                request_outputs=outputs
            )
            experiment_output.save()
            print(f"Saved {experiment_output.id}")