import numpy as np
import os
import random

from transformers import AutoTokenizer

from llmperf.config.models import get_model_by_name
from llmperf.config.workloads import Request, Workload
from llmperf.config.workloads import get_workload_by_name
from llmperf.constants import WORKLOADS_DIR

GAMMA_PARAMS = {
        ## ! Keep shape = 0.5 based on BurstGPT paper
        # Coefficient of Variation (CV) = 1 / sqrt(shape), e.g CV = 8, A = 1/64
        # request rate: (shape, scale) (A, B)
        10.0: (0.5, 0.197),
        9.5: (0.5, 0.211),
        9.0: (0.5, 0.223),
        8.5: (0.5, 0.233),
        8.0: (0.5, 0.248),
        7.5: (0.5, 0.263),
        7.0: (0.5, 0.282),
        6.5: (0.5, 0.301),
        6.0: (0.5, 0.336),
        5.5: (0.5, 0.37),
        5.0: (0.5, 0.403),
        4.5: (0.5, 0.442),
        4.0: (0.5, 0.487),
        3.5: (0.5, 0.56),
        3.0: (0.5, 0.66),
        2.5: (0.5, 0.79),
        2.0: (0.5, 0.95),
        1.5: (0.5, 1.3),
        1.0: (0.5, 2.0),
        0.5: (0.5, 4),
        0.1: (0.5, 20.0)
    }

if __name__ == '__main__':
    """
    This script creates text and mixed modalities workloads for every request rate in REQUEST_RATES
    It creates mixed modalities workloads by substituting the top x% of requests based on the REPLACEMENT_PCTS
    """
    REPLACEMENT_PCTS = [0.15, 0.30, 0.45]
    REQUEST_RATES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
    REFERENCE_MODEL_NAME = "Mistral-7b"

    SEED = 0
    DURATION = 100

    for REQUEST_RATE in REQUEST_RATES:
        random.seed(SEED)
        np.random.seed(SEED)
        
        GAMMA_SHAPE = GAMMA_PARAMS[REQUEST_RATE][0]
        GAMMA_SCALE = GAMMA_PARAMS[REQUEST_RATE][1]
        
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
        timestamps = []
        timestamp = 0.0
        while timestamp < DURATION:
            delta_time = np.random.gamma(GAMMA_SHAPE, GAMMA_SCALE)
            timestamp = delta_time + timestamp
            timestamps.append(timestamp)

        # If there are more requests than needed, remove unnecessary requests
        if len(timestamps) < len(requests):
            requests = requests[:len(timestamps)]
        else:
            # If there are less requests than needed, cycle through requests
            extended_requests = [requests[i % len(requests)] for i in range(len(timestamps))]
            requests[:] = extended_requests
        
        assert len(timestamps) == len(requests)

        name = f"{text_workload_name} with Gamma {REQUEST_RATE}"
        alias = f"text-gamma-{REQUEST_RATE}"
        text_gamma_workload = Workload(
            name=name,
            path=os.path.join(WORKLOADS_DIR, "text-gamma"),
            alias=alias,
            modalities="text",
            modality_pct=1.0,
            arrival_dist="gamma",
            requests=requests,
            timestamps=timestamps
        )
        text_gamma_workload.save()

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
            ## largest_requests_idxs.sort()
            
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

            name = f"Mixed Modalities with Gamma {REQUEST_RATE} {int(REPLACEMENT_PCT*100)}%"
            alias = f"mix-gamma-{REQUEST_RATE}-{int(REPLACEMENT_PCT*100)}"
            mix_gamma_workload = Workload(
                name=name,
                path=os.path.join(WORKLOADS_DIR, f"mix-gamma-{int(REPLACEMENT_PCT*100)}"),
                alias=alias,
                modalities=["text", "image", "video", "audio"],
                modality_pct=[1-REPLACEMENT_PCT, REPLACEMENT_PCT/3, REPLACEMENT_PCT/3, REPLACEMENT_PCT/3],
                modality_dist="uniform",
                arrival_dist="gamma",
                requests=requests,
                timestamps=timestamps
            )
            mix_gamma_workload.save()