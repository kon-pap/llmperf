import numpy as np
import os
import random

from transformers import AutoTokenizer

from llmperf.config.workloads import Request, Workload
from llmperf.config.workloads import get_workload_by_name
from llmperf.constants import WORKLOADS_DIR

def categorize_numbers(n, percentages=(0.6, 0.3, 0.1)):
    if sum(percentages) != 1.0:
        raise ValueError("Percentages must sum to 1.0")
    
    numbers = list(range(n))
    random.shuffle(numbers)
    
    a_count = int(n * percentages[0])
    b_count = int(n * percentages[1])
    
    sand = numbers[:a_count]
    pebbles = numbers[a_count:a_count + b_count]
    rocks = numbers[a_count + b_count:]
    
    return sand, pebbles, rocks


if __name__ == '__main__':
    """
    This script creates mixed modalities workloads for every request rate in REQUEST_RATES
    It creates mixed modalities workloads by randomly replacing text requests.
    RPS_PCTS = [% of sand, % of pebbles, % of rocks]
    """
    RPS_PCTS = [0.45, 0.35, 0.2] # [0.6, 0.3, 0.1] [0.7, 0.3, 0.0]
    REQUEST_RATES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    REFERENCE_MODEL_NAME = "Mistral-7b"

    SEED = 0
    TIME_QUANTUM = 10
    DURATION = 100

    
    for REQUEST_RATE in REQUEST_RATES:
        random.seed(SEED)
        np.random.seed(SEED)

        sand_workload_name = "Text Conversations"
        pebbles_workload_name = "Image Reasoning"
        rocks_workload_name = "Video Description"

        # Get text requests
        sand_workload = get_workload_by_name(sand_workload_name)
        sand_workload.load()
        requests = sand_workload.requests
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

        name = f"{sand_workload_name} with Poisson {REQUEST_RATE} (Sand Only)"
        alias = f"sand-poisson-{REQUEST_RATE}"
        sand_workload = Workload(
            name=name,
            path=os.path.join(WORKLOADS_DIR, f"rps-poisson-{'-'.join([str(int(x*100)) for x in RPS_PCTS])}"),
            alias=alias,
            modalities="text",
            modality_pct=1.0,
            arrival_dist="poisson",
            requests=requests,
            timestamps=timestamps
        )
        sand_workload.save()

        # Get idxs of requests to be replaced
        sand_pct = RPS_PCTS[0]
        pebbles_pct = RPS_PCTS[1]
        rocks_pct = RPS_PCTS[2]

        _, pebbles_idxs, rocks_idxs = categorize_numbers(len(requests), (sand_pct, pebbles_pct, rocks_pct))

        if pebbles_pct == 0.0:
            pebbles_idxs = []
        
        if rocks_pct == 0.0:
            rocks_idxs = []

        # Replace pebbles
        pebbles_workload = get_workload_by_name(pebbles_workload_name)
        pebbles_workload.load()
        pebble_requests = pebbles_workload.requests[:len(pebbles_idxs)]
        random.shuffle(pebble_requests)

        for idx, request in zip(pebbles_idxs, pebble_requests):
            requests[idx] = Request(**vars(request))
        
        # Replace rocks
        rocks_workload = get_workload_by_name(rocks_workload_name)
        rocks_workload.load()
        rock_requests = rocks_workload.requests[:len(rocks_idxs)]
        random.shuffle(rock_requests)

        for idx, request in zip(rocks_idxs, rock_requests):
            requests[idx] = Request(**vars(request))

        name = f"Rock - Pebbles - Sand with Poisson {REQUEST_RATE} {'%-'.join([str(int(x*100)) for x in RPS_PCTS])}%"
        alias = f"rps-poisson-{REQUEST_RATE}-{'-'.join([str(int(x*100)) for x in RPS_PCTS])}"
        rps_poisson_workload = Workload(
            name=name,
            path=os.path.join(WORKLOADS_DIR, f"rps-poisson-{'-'.join([str(int(x*100)) for x in RPS_PCTS])}"),
            alias=alias,
            modalities=["text", "image", "video"],
            modality_pct=RPS_PCTS,
            modality_dist="categorical",
            arrival_dist="poisson",
            requests=requests,
            timestamps=timestamps
        )
        rps_poisson_workload.save()