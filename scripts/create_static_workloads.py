import os

from llmperf.constants import WORKLOADS_DIR
from llmperf.config.datasets import get_dataset_by_name
from llmperf.config.workloads import Request, Workload

if __name__ == '__main__':
    names = [
        "Text Conversations",
        "Image Reasoning",
        "Video Description",
        "Audio Captioning"
    ]
    aliases = [
        "text-static",
        "image-static",
        "video-static",
        "audio-static"
    ]

    for name, alias in zip(names, aliases):
        dataset = get_dataset_by_name(name)
        data = dataset.load()

        requests: Request = []
        ids = []
        for record in data:
            prompt = dataset.get_input() if name == "Audio Captioning" else dataset.get_input(record)
            response = dataset.get_output(record)
            modality_path = dataset.get_modality_path(record) if name != "Text Conversations" else None
            modality_size = dataset.get_modality_size(record) if name != "Text Conversations" else None
            
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

        static_workload = Workload(
            name=name,
            path=os.path.join(WORKLOADS_DIR, "static"),
            alias=alias,
            modalities=alias.split("-")[0],
            modality_pct=1.0,
            requests=requests
        )
        static_workload.save()