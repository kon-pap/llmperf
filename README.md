# LLMPerf Inference Benchmark Suite

LLMPerf Inference is a benchmark suite for measuring how fast systems can run MLLMs in a variety of deployment scenarios.

## Installation

To install pyenv:
```
curl -fsSL https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile

echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

exec "$SHELL"
```
1. Clone repositories
    ```
    git clone git@gitlab.software.imdea.org:muse-lab/mllm-inference-workload-eval.git
    cd mllm-inference-workload-eval/
    git clone git@gitlab.software.imdea.org:muse-lab/vllm.git
    cd vllm/
    git checkout dev
    ```
2. Create virtual environemnt
    ```
    pyenv install 3.12.8
    pyenv virtualenv 3.12.8 vllm-v0.7.2
    pyenv activate vllm-v0.7.2
    ```
3. Install vllm (for image, video, audio)
    ```
    VLLM_USE_PRECOMPILED=1 pip install --editable .
    ```
    or (for v0.8.4)
    ```
    export VLLM_COMMIT=dc1b4a6f1300003ae27f033afbdff5e2683721ce
    export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
    pip install --editable .
    ```
4. Install llmperf
    ```
    cd ..
    pip install --editable .
    ```

Remember to check the contributing guide for vLLM!
```
cd vllm
pip install -r requirements/dev.txt
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files
```

## Datasets

Donwload datasets separately or download a minimal version provided by us

## Workloads

```
python scripts/create_static_workloads.py # Create {text,image,video,audio}-static workloads from the datasets
python scripts/create_poisson_workloads.py # Create {text,mix}-poisson workloads using static workloads
python scripts/create_gamma_workloads.py # Create {text,mix}-gamma workloads using static workloads
python scripts/create_rps_workloads.py # Create rocks, pebbles, sand workloads using static workloads
python scripts/create_multi_stream_workloads.py --workloads text-static image-static video-static --request-rates 0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0 5.0 6.0 8.0 10.0 # Create multi modalities workloads
```

## Experiments

```
python scripts/run_static_workloads.py # Run static workloads in isolation
python scripts/run_vllm_workloads.py # Run workloads in vllm
python scripts/run_vllm_chunk_workloads.py # Run workloads in vllm with chunked prefill
python scripts/run_mem_balloon_workloads.py # Run workloads in vllm with memory ballooning
```