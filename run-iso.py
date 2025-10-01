import os
import subprocess
from pathlib import Path

from llmperf.constants import EXPERIMENTS_LOG

# --- Setup ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

script = "scripts/run_static_workloads.py"
logfile = Path(EXPERIMENTS_LOG)
logfile.parent.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def log(msg: str):
    with logfile.open("a") as f:
        f.write(msg + "\n")

def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(["python", script] + cmd, check=True)

# --- Git checkout baseline ---
subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.1.baseline"], check=True)
log("# vllm.baseline")

# --- Workloads definition ---
workloads = {
    "llava-ov-small": [
        {"workload": "text-static"},
        {"workload": "image-static"},
        {"workload": "video-static"},
    ],
    "llava-ov": [
        {"workload": "text-static"},
        {"workload": "image-static"},
        {"workload": "video-static"},
        {"workload": "text-static-long"}
    ],
    "gemma-3-small": [
        {"workload": "text-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "image-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "video-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048, "multi-image": True},
    ],
    "gemma-3": [
        {"workload": "text-static", "max-model-len": 16384, "max-num-batched-tokens": 16384, "num-gpu-blocks-override": 1024},
        {"workload": "image-static", "max-model-len": 16384, "max-num-batched-tokens": 16384, "num-gpu-blocks-override": 1024},
        {"workload": "video-static", "max-model-len": 18432, "max-num-batched-tokens": 16384, "num-gpu-blocks-override": 1152, "multi-image": True, "gpu-util": 0.98},
    ],
    "qwen-2.5-small": [
        {"workload": "text-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "image-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "video-static", "max-model-len": 128000, "max-num-batched-tokens": 128000, "num-gpu-blocks-override": 8000},
    ],
    "qwen-2.5": [
        {"workload": "text-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "image-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "video-static", "max-model-len": 82112, "max-num-batched-tokens": 80970, "num-gpu-blocks-override": 5132},
    ],
    "pixtral": [
        {"workload": "text-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "image-static", "max-model-len": 32768, "max-num-batched-tokens": 32768, "num-gpu-blocks-override": 2048},
        {"workload": "video-static", "max-model-len": 45056, "max-num-batched-tokens": 45056, "num-gpu-blocks-override": 2816, "multi-image": True},
    ],
}

# --- Run workloads ---
for model, configs in workloads.items():
    log(f"## {model}")
    for cfg in configs:
        args = ["--model", model]
        for k, v in cfg.items():
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            else:
                args.extend([f"--{k}", str(v)])
        run(args)
