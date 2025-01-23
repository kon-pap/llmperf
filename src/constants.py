import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))

MODELS_DIR = "/srv/muse-lab/models" # os.path.join(ROOT_DIR, "models")
DATASETS_DIR = "/srv/muse-lab/datasets" # os.path.join(ROOT_DIR, "datasets")

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
EXPERIMENTS_OUTPUTS_DIR = os.path.join(ARTIFACTS_DIR, "outputs")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
WORKLOADS_DIR = os.path.join(ARTIFACTS_DIR, "workloads")
EXPERIMENTS_LOG = os.path.join(ARTIFACTS_DIR, "experiments-log.jsonl")