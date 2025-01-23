from setuptools import setup, find_packages

setup(
    name="llmperf",
    version="0.0.1",
    author="Konstantinos Papaioannou, Giannis Dalianis",
    author_email="konstantinos.papaioannou@imdea.org, ioannis.dalianis@imdea.org",
    description="LLMPerf Inference is a benchmark suite for measuring how fast systems can run MLLMs in a variety of deployment scenarios.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.software.imdea.org/muse-lab/mllm-inference-workload-eval",
    packages=find_packages("src"),
    package_dir={"": "src"}, 
    install_requires=[
        "ffmpeg-python>=0.2.0",
        "pillow>=10.4.0"
    ],
    python_requires=">=3.9",
)