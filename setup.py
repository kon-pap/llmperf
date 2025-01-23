from setuptools import setup, find_packages

setup(
    name="llmperf",
    version="0.0.1",
    author="Konstantinos Papaioannou, Giannis Dalianis",
    author_email="konstantinos.papaioannou@imdea.org, ioannis.dalianis@imdea.org",
    description="LLMPerf Inference is a benchmark suite for measuring how fast systems can run MLLMs in a variety of deployment scenarios.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-repo",
    packages=find_packages("src"),
    package_dir={"": "src"}, 
    install_requires=[
    ],
    python_requires=">=3.9",
)