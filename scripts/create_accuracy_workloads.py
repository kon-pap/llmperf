import os

from llmperf.constants import WORKLOADS_DIR
from llmperf.config.datasets import get_dataset_by_name
from llmperf.config.workloads import Request, Workload

if __name__ == '__main__':
    names = [
        "Multiple Choice (MMBench)",
        "Q&A (LLaVABench)",
        "Captioning (COCO-Val)",
        "Multiple Choice (Video-MME)",
        "Q&A (MMBench-Video)",
        "Captioning (TempCompass)"
    ]

    for name in names:
        dataset = get_dataset_by_name(name)
        data = dataset.load()

        modality = "image" if name in ["Multiple Choice (MMBench)", "Q&A (LLaVABench)", "Captioning (COCO-Val)"] else "video"

        requests: Request = []
        ids = []
        for record in data:
            prompt = dataset.get_input() if name == "Captioning (COCO-Val)" else dataset.get_input(record)
            response = dataset.get_output(record)
            modality_path = dataset.get_modality_path(record)
            modality_size = dataset.get_modality_size(record)
            
            if prompt and response:
                request_args = {
                    "input": prompt,
                    "output": response,
                }

                if modality_path:
                    request_args["modality_path"] = modality_path
                if modality_size:
                    request_args["modality_size"] = modality_size
                
                request = Request(**request_args)
                if request.id not in ids:
                    requests.append(request)
                    ids.append(request.id)

        workload = Workload(
            name=name,
            path=os.path.join(WORKLOADS_DIR, "static"),
            alias=dataset.alias,
            modalities=modality,
            modality_pct=1.0,
            requests=requests
        )
        workload.save()