# LLMPerf Inference Benchmark Suite

LLMPerf Inference is a benchmark suite for measuring how fast systems can run MLLMs in a variety of deployment scenarios.

## Installation

1. Clone recursively
2. Create virtual environemnt
3. Install vllm (for image, video, audio)
4. Install llmperf

## Datasets

Donwload datasets separately or download a minimal version provided by us

## Workloads

```
python src/scripts/create_static_workloads.py # Create {text,image,video,audio}-static workloads from the datasets
python src/scripts/create_poisson_workloads.py # Create {text,mix}-poisson workloads using static workloads
python src/scripts/create_gamma_workloads.py # Create {text,mix}-gamma workloads using static workloads
```

## Experiments

```
python src/scripts/run_static_workloads.py # Run static workloads in isolation
python src/scripts/run_vllm_workloads.py # Run workloads in vllm
python src/scripts/run_vllm_chunk_workloads.py # Run workloads in vllm with chunked prefill
python src/scripts/run_mem_balloon_workloads.py # Run workloads in vllm with memory ballooning
```