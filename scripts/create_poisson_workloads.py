import numpy as np
import os
import random

from transformers import AutoTokenizer

from llmperf.config.models import get_model_by_name
from llmperf.config.workloads import Request, Workload
from llmperf.config.workloads import get_workload_by_name
from llmperf.constants import WORKLOADS_DIR

if __name__ == '__main__':
    """
    This script creates text and mixed modalities workloads for every request rate in REQUEST_RATES
    It creates mixed modalities workloads by substituting the top x% of requests based on the REPLACEMENT_PCTS
    """
    REPLACEMENT_PCTS = [0.15, 0.30, 0.45]
    REQUEST_RATES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    REFERENCE_MODEL_NAME = "Mistral-7b"

    SEED = 0
    TIME_QUANTUM = 10
    DURATION = 100

    
    for REQUEST_RATE in REQUEST_RATES:
        random.seed(SEED)
        np.random.seed(SEED)

        text_workload_name = "Text Conversations"
        ## ! Replacement follow the list below. First it replaces using requests from the first element of the list, then the second one, etc.
        modality_workload_names = [
            "Video Description",
            "Image Reasoning",
            "Audio Captioning"
        ]

        # Get text requests
        text_workload = get_workload_by_name(text_workload_name)
        text_workload.load()
        requests = text_workload.requests
        random.shuffle(requests)

        # Generate timestamps
        lam = REQUEST_RATE * (TIME_QUANTUM / 1000)
        quantums_per_sec = 1000 / TIME_QUANTUM
        arrival_times = np.random.poisson(
            lam=lam, size=int(DURATION * quantums_per_sec))
        timestamps = []
        for i, n in enumerate(arrival_times):
            timestamps += [i * (TIME_QUANTUM / 1000)] * n

        # If there are more requests than needed, remove unnecessary requests
        if len(timestamps) < len(requests):
            requests = requests[:len(timestamps)]
        else:
            # If there are less requests than needed, cycle through requests
            extended_requests = [requests[i % len(requests)] for i in range(len(timestamps))]
            requests[:] = extended_requests
        
        assert len(timestamps) == len(requests)

        name = f"{text_workload_name} with Poisson {REQUEST_RATE}"
        alias = f"text-poisson-{REQUEST_RATE}"
        text_poisson_workload = Workload(
            name=name,
            path=os.path.join(WORKLOADS_DIR, "text-poisson"),
            alias=alias,
            modalities="text",
            modality_pct=1.0,
            arrival_dist="poisson",
            requests=requests,
            timestamps=timestamps
        )
        text_poisson_workload.save()

        for REPLACEMENT_PCT in REPLACEMENT_PCTS:
            # Get idxs of requests to be replaced
            model = get_model_by_name(REFERENCE_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(model.path)

            request_input_lengths = {}
            for idx, request in enumerate(requests):
                input_length = len(tokenizer.encode(request.input)[1:])
                request_input_lengths[idx] = input_length

            sorted_request_input_lengths = sorted(request_input_lengths.items(), key=lambda item: item[1], reverse=True)
            top_n = max(1, int(len(sorted_request_input_lengths) * REPLACEMENT_PCT))
            largest_requests_idxs = [item[0] for item in sorted_request_input_lengths[:top_n]]
            
            # Get modality requests
            requests_per_modality = []
            for workload_name in modality_workload_names:
                modality_workload = get_workload_by_name(workload_name)
                modality_workload.load()
                requests_per_modality.append(modality_workload.requests)

            for modality_requests in requests_per_modality:
                random.shuffle(modality_requests)
            
            num_requests_per_modality = int(len(largest_requests_idxs) / len(requests_per_modality))
            mixed_requests = [request for modality_requests in requests_per_modality for request in modality_requests[:num_requests_per_modality]]
            
            if len(largest_requests_idxs) > len(mixed_requests):
                extra = len(largest_requests_idxs) - len(mixed_requests)
                mixed_requests += requests_per_modality[-1][num_requests_per_modality:num_requests_per_modality+extra]

            assert len(largest_requests_idxs) == len(mixed_requests)
            
            for idx, request in zip(largest_requests_idxs, mixed_requests):
                requests[idx] = Request(**vars(request))

            name = f"Mixed Modalities with Poisson {REQUEST_RATE} {int(REPLACEMENT_PCT*100)}%"
            alias = f"mix-poisson-{REQUEST_RATE}-{int(REPLACEMENT_PCT*100)}"
            mix_poisson_workload = Workload(
                name=name,
                path=os.path.join(WORKLOADS_DIR, f"mix-poisson-{int(REPLACEMENT_PCT*100)}"),
                alias=alias,
                modalities=["text", "image", "video", "audio"],
                modality_pct=[1-REPLACEMENT_PCT, REPLACEMENT_PCT/3, REPLACEMENT_PCT/3, REPLACEMENT_PCT/3],
                modality_dist="uniform",
                arrival_dist="poisson",
                requests=requests,
                timestamps=timestamps
            )
            mix_poisson_workload.save()