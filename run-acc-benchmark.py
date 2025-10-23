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
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(["python", script] + cmd, check=True)

# --- Git checkout baseline ---
subprocess.run(["git", "-C", "vllm", "checkout", "v0.0.1.baseline"], check=True)
log("# vllm.baseline")

pixtral_fields = {
    "max-model-len": 2750*16, "max-num-batched-tokens": 2750*16, "num-gpu-blocks-override": 2750
}

# --- Workloads definition ---
# --- Image workloads ---
img_workload_aliases = ["mmbench-mc", "llavabench-qna", "cocoval-captioning"]
compression_ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

image_workloads = [
    {"workload": wl, "compression-ratio": ratio}
    for wl in img_workload_aliases
    for ratio in compression_ratios
]

workloads = {
    "llava-ov-small": image_workloads,
    "llava-ov": image_workloads,
    "qwen-2-small": image_workloads,
    "qwen-2": image_workloads,
    "pixtral": [{**w, **pixtral_fields} for w in image_workloads]
}

# --- Video workloads ---
vid_workload_aliases = ["videomme-mc", "mmbench-video-qna", "tempcompass-captioning"]
strategies = ["uniform", "scene_change", "sharpness_based", "motion_based"]
num_frames = [4, 8, 16, 32, 64]

video_workloads = [
    {"workload": wl, "strategy": strategy, "num_frames": nf}
    for wl in vid_workload_aliases
    for strategy in strategies
    for nf in num_frames
]

workloads = {
    "llava-ov-small": video_workloads,
    "llava-ov": video_workloads,
    "qwen-2-small": [{**w, "smart-resize": True} for w in video_workloads],
    "qwen-2": [{**w, "smart-resize": True} for w in video_workloads],
    "pixtral": [{**w, **pixtral_fields, "smart-resize": True, "multi-image": True} for w in video_workloads]
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
