import argparse
import time

from datetime import datetime
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

from llmperf.config.approaches import get_approach_by_alias
from llmperf.config.models import get_model_by_alias
from llmperf.config.workloads import get_workload_by_alias
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import get_modality_token_index, prepare_final_prompt

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

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

    approach = get_approach_by_alias("iso")
    model = get_model_by_alias(args.model)
    workload = get_workload_by_alias(args.workload)
    workload.load()

    try:
        llm = LLM(
            model=model.path,
            gpu_memory_utilization=args.gpu_util,
            swap_space=args.swap_space,
            scheduling_policy=approach.scheduling_policy,
            disable_log_stats=False,
            max_model_len=model.max_model_len,
            max_num_batched_tokens=model.max_model_len,
            num_gpu_blocks_override=model.max_model_len // 16
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

            modality_token_index = get_modality_token_index(request, model)
            final_prompt = prepare_final_prompt(request, model)

            req_output = llm.generate(
                prompts=[final_prompt],
                sampling_params=sampling_params,
                use_tqdm=False
            )[0]

            now = time.time()
            outputs.append(
                RequestOutput.from_vllm_output(
                    request.id, req_output, modality_token_index
                )
            )

    finally:
        elapsed_time = now - start_time

        experiment_output = ExperimentOutput(
            id=f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}",
            elapsed_time=elapsed_time,
            request_outputs=outputs
        )
        experiment_output.save()
        print(f"Saved {experiment_output.id}")