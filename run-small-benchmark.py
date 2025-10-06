import os
import subprocess
import time
from pathlib import Path

from llmperf.constants import EXPERIMENTS_LOG

# --- Environment setup ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

script = "scripts/run_vllm_workloads.py"
logfile = Path(EXPERIMENTS_LOG)
logfile.parent.mkdir(parents=True, exist_ok=True)

# --- Benchmark configuration ---
MODEL = "llava-ov"

REQUEST_RATES = [1.0, 1.5, 2.0]

WORKLOADS = [
    "text-only",
    "long-text-light",
    "image-light",
    "video-light",
    "long-text-heavy",
    "image-heavy",
    "video-heavy",
    "mixed-light",
    "mixed-heavy"
]

PROFILING_DATA = [
    "text-static-llava-ov-iso-20250514-192704",
    "image-static-llava-ov-iso-20250514-204328",
    "video-static-llava-ov-iso-20250515-235113",
    "text-static-long__llava-ov__iso__v0.0.1.naive-slo-aware__20250910-165223__maxlendef__batchdef__blocksdef__gpu0.95__swap0"
]

# --- Helpers ---
def format_hms(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def log(msg: str):
    with logfile.open("a") as f:
        f.write(msg + "\n")

def run(args: list[str]):
    print("Running:", " ".join(args), flush=True)
    try:
        subprocess.run(["python", script] + args, check=True)
    except subprocess.CalledProcessError as e:
        log(f"### Failed {args[3]}")
        print(f"❌ Failed: {' '.join(args)} (exit {e.returncode}), continuing...", flush=True)
    except Exception as e:
        log(f"### Failed {args[3]}")
        print(f"⚠️ Unexpected error: {e}, continuing...", flush=True)

# --- Single experiment block ---
def run_benchmark(slo_aware: float  = False, num_gpu_blocks: int = None, approach: str = "vllm"):
    for w in WORKLOADS:
        for rr in REQUEST_RATES:
            workload = f"{w}-{rr}"
            args = [
                "--model", MODEL,
                "--workload", workload,
                "--approach", approach,
                "--profiling-data", *PROFILING_DATA,
            ]
            if slo_aware:
                args += ["--slo-aware"]
            if num_gpu_blocks is not None:
                args += ["--num-gpu-blocks-override", str(num_gpu_blocks)]
                args += ["--max-num-encoder-input-tokens", str(num_gpu_blocks * 16)]
                args += ["--num-encoder-tokens-override", str(num_gpu_blocks * 16)]
            else:
                args += ["--max-num-encoder-input-tokens", "32768"]
                args += ["--num-encoder-tokens-override", "32768"]
            run(args)

# --- Run benchmark ---
if __name__ == "__main__":
    iteration_times = []
    total_start = time.time()

    for num_gpu_blocks in [None, 2048, 1024, 512, 256]:
        iter_start = time.time()

        subprocess.run(["git", "-C", "vllm", "stash", "push"], check=True)

        blocks_msg = f"# {num_gpu_blocks} Blocks" if num_gpu_blocks else "# Max Memory"
        log(blocks_msg)
        
        subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.1.baseline"], check=True)
        log(f"## vllm.baseline")
        run_benchmark(num_gpu_blocks=num_gpu_blocks)
        
        subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.1.naive-classifier"], check=True)
        log(f"## vllm.naive-classifier")
        run_benchmark(num_gpu_blocks=num_gpu_blocks)
        
        subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.1.naive-slo-aware"], check=True)
        log(f"## vllm.naive-slo-aware")
        run_benchmark(slo_aware=True, num_gpu_blocks=num_gpu_blocks)

        subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.2.naive-slo-aware"], check=True)
        log(f"## vllm.naive-ttft-slo-aware")
        run_benchmark(slo_aware=True, num_gpu_blocks=num_gpu_blocks)

        subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.1.naive-mlq"], check=True)
        log(f"## vllm.naive-mlq")
        run_benchmark(slo_aware=True, num_gpu_blocks=num_gpu_blocks)

        subprocess.run(["git", "-C", "vllm", "stash", "pop"], check=True)
        subprocess.run(["git", "-C", "vllm", "checkout", "dev-v1-konpap"], check=True)
        log(f"## vllm.wip")
        run_benchmark(slo_aware=True, num_gpu_blocks=num_gpu_blocks)

        iter_end = time.time()
        iter_duration = iter_end - iter_start
        iteration_times.append((blocks_msg, iter_duration))
    
    total_end = time.time()
    total_duration = total_end - total_start

    log("### Benchmark Timing Report ###")
    for blocks_msg, duration in iteration_times:
        log(f"### \t-{blocks_msg[1:]}: {format_hms(duration)}")
    log(f"### Total execution time: {format_hms(total_duration)} ###")
