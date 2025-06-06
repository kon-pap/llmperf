import argparse
import asyncio
import time
import yaml

from datetime import datetime
from tqdm.asyncio import tqdm_asyncio

from vllm import AsyncEngineArgs, RequestOutput as vllmRequestOutput, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from llmperf.config.approaches import get_approach_by_alias
from llmperf.config.models import get_model_by_alias
from llmperf.config.workloads import get_workload_by_alias, Request
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import get_modality_token_index, prepare_final_prompt

# Global variables
llm = None
model = None

async def send_request(idx: int, request: Request, timestamp: float, max_tokens: int) -> vllmRequestOutput:
    await asyncio.sleep(timestamp)

    sampling_params = SamplingParams(
        ignore_eos=True,
        max_tokens=max_tokens
    )

    final_prompt = prepare_final_prompt(request, model)
    
    final_output = None
    async for output in llm.generate(final_prompt, sampling_params, str(idx)):
        final_output = output

    assert final_output is not None

    return final_output

async def main(args: argparse.Namespace):
    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

    approach = get_approach_by_alias(args.approach)
    global model
    model = get_model_by_alias(args.model)
    workload = get_workload_by_alias(args.workload)
    workload.load()

    profiling_data = {}
    for eo_id in args.profiling_data:
        eo = ExperimentOutput(id=eo_id)
        eo.load()

        for o in eo.request_outputs:
            profiling_data[o.id] = o

    try:
        engine_args = AsyncEngineArgs(
            model=model.path,
            gpu_memory_utilization=args.gpu_util,
            swap_space=args.swap_space,
            scheduling_policy=approach.scheduling_policy,
            disable_log_requests=True,
            disable_log_stats = False,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            num_gpu_blocks_override=args.num_gpu_blocks_override
        )

        global llm
        llm = AsyncLLM.from_engine_args(engine_args, log_file=args.log_file)

        requests = workload.requests
        timestamps = workload.timestamps

        # Match indexes to request
        idx_to_req = {}
        for idx, request in enumerate(requests):
            idx_to_req[str(idx)] = request

        # Create max tokens list
        max_tokens = []
        for request in requests:
            mt = profiling_data[request.id].decode_tokens_cnt
            max_tokens.append(mt)
        
        outputs = []
        start_time = now = time.time()
        pending_requests = [
            send_request(idx, request, ts, mt) for idx, (request, ts, mt) in enumerate(zip(requests, timestamps, max_tokens))
        ]

        request_outputs = []
        for completed_req_output in tqdm_asyncio.as_completed(pending_requests):
            req_output: vllmRequestOutput = await completed_req_output
            request_outputs.append(req_output)
            now = time.time()

        for req_output in request_outputs:
            original_request = idx_to_req[req_output.request_id]
            modality_token_index = get_modality_token_index(original_request, model)

            outputs.append(
                RequestOutput.from_vllm_output(
                    original_request.id, req_output, modality_token_index
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
        print(f"Saved {experiment_output.id} outputs")

        experiment_output.save_engine_stats(log_file=args.log_file)
        print(f"Saved {experiment_output.id} engine stats")

def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to a YAML config file")
    config_args, remaining = config_parser.parse_known_args()
    
    parser = argparse.ArgumentParser(
        description="Parse resource usage and experiment metadata."
    )

    parser.add_argument("--gpu-util", type=float, default=0.95,
                        help="GPU utilization percentage (default: 0.95)")
    parser.add_argument("--swap-space", type=int, default=0,
                        help="Swap space used in GB (default: 0)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model alias (e.g., llava-ov)")
    parser.add_argument("--workload", type=str, required=True,
                        help="Workload alias (e.g., text-poisson-1.0)")
    parser.add_argument("--approach", type=str, required=True,
                        help="Approach alias (e.g., vllm)")
    
    parser.add_argument("--log-file", type=str, default="engine-stats.log",
                        help="Log file path (default: engine-stats.log)")
    
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Model context length")
    parser.add_argument("--max-num-batched-tokens", type=str, default=None,
                        help="Maximum number of batched tokens per iteration")
    parser.add_argument("--num-gpu-blocks-override", type=str, default=None,
                        help="Number of GPU blocks")
    
    parser.add_argument("--profiling-data", nargs="+", type=str, required=True,
                        help="List of experiment output ids")
    
    final_args = []

    if config_args.config:
        with open(config_args.config, 'r') as f:
            yaml_args = yaml.safe_load(f)

        for k, v in yaml_args.items():
            key = f"--{k.replace('_', '-')}"
            if isinstance(v, list):
                final_args.append(key)
                final_args.extend(map(str, v))
            elif v is not None:
                final_args.extend([key, str(v)])

    final_args += remaining

    return parser.parse_args(final_args)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))   