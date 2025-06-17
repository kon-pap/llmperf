import argparse
import numpy as np
import os
import random

from llmperf.config.workloads import Workload
from llmperf.config.workloads import get_workload_by_alias
from llmperf.constants import WORKLOADS_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse workload metadata."
    )
    
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for randomness (default: 0)')
    parser.add_argument('--time-quantum', type=int, default=10,
                        help='Time quantum in ms (default: 10)')
    parser.add_argument('--duration', type=int, default=100,
                        help='Duration of workload in s (default: 100)')
    
    parser.add_argument('--workloads', nargs="+", type=str, required=True,
                        help='Workload aliases (e.g., video-static)')
    
    parser.add_argument("--request-rates", nargs="+", type=float, required=True,
                        help="List of request rates")

    return parser.parse_args()

if __name__ == '__main__':
    """
    This script creates multi modalities workloads for every request rate in REQUEST_RATES
    It creates multi modalities workloads by creating one request stream per modality.
    """
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    final_workloads = {}
    for workload_alias in args.workloads:
        workload = get_workload_by_alias(workload_alias)
        workload.load()

        final_workloads[workload_alias] = {}
        
        requests = workload.requests
        random.shuffle(requests)

        for req_rate in args.request_rates:
            # Generate timestamps
            lam = req_rate * (args.time_quantum / 1000)
            quantums_per_sec = 1000 / args.time_quantum
            arrival_times = np.random.poisson(
                lam=lam, size=int(args.duration * quantums_per_sec))
            timestamps = []
            for i, n in enumerate(arrival_times):
                timestamps += [i * (args.time_quantum / 1000)] * n

            # If there are more requests than needed, remove unnecessary requests
            final_requests = requests
            if len(timestamps) < len(final_requests):
                final_requests = final_requests[:len(timestamps)]
            else:
                # If there are less requests than needed, cycle through requests
                extended_requests = [final_requests[i % len(final_requests)] for i in range(len(timestamps))]
                final_requests[:] = extended_requests
        
            assert len(timestamps) == len(final_requests)

            name = f"{workload.name} with Poisson {req_rate} v2"
            alias = f"{workload_alias}-poisson-{req_rate}-v2"
            final_workload = Workload(
                name=name,
                path=os.path.join(WORKLOADS_DIR, f"multi-stream"),
                alias=alias,
                modalities=workload.modalities,
                modality_pct=1.0,
                arrival_dist="poisson",
                requests=final_requests,
                timestamps=timestamps
            )
            final_workload.save()

            final_workloads[workload_alias][req_rate] = final_workload

    max_sum = max(args.request_rates)

    first_limit = 1.0
    first_filtered_req_rates = [i for i in args.request_rates if i <= first_limit]
    
    combinations = []
    for f in first_filtered_req_rates:
        for b in args.request_rates:
            if f + b < max_sum:
                combinations.append((b, f))

                base_workload_alias = args.workloads[0]
                w1 = final_workloads[base_workload_alias][b]

                first_workload_alias = args.workloads[1]
                w2 = final_workloads[first_workload_alias][f]

                w1_combined = list(zip(w1.timestamps, w1.requests))
                w2_combined = list(zip(w2.timestamps, w2.requests))

                merged = w1_combined + w2_combined
                merged_sorted = sorted(merged, key=lambda x: x[0])
                
                final_timestamps, final_requests = zip(*merged_sorted)

                modalities = [w1.modalities, w2.modalities]
                modality_pct = [len(w1.requests)/ (len(w1.requests) + len(w2.requests)), len(w2.requests)/ (len(w1.requests) + len(w2.requests))]

                name = f"{base_workload_alias[0].upper()}{first_workload_alias[0].upper()} with Poisson {b}-{f} v2"
                alias = f"{base_workload_alias[0]}{first_workload_alias[0]}-poisson-{b}-{f}-v2"
                final_workload = Workload(
                    name=name,
                    path=os.path.join(WORKLOADS_DIR, f"multi-stream"),
                    alias=alias,
                    modalities=modalities,
                    modality_pct=modality_pct,
                    modality_dist="categorical",
                    arrival_dist="poisson",
                    requests=final_requests,
                    timestamps=final_timestamps
                )
                final_workload.save()


    second_limit = 0.5
    second_filtered_req_rates = [i for i in args.request_rates if i <= second_limit]

    combinations = []
    for s in second_filtered_req_rates:
        for f in first_filtered_req_rates:
            for b in args.request_rates:
                if s + f + b < max_sum:
                    combinations.append((b, f, s))

                    base_workload_alias = args.workloads[0]
                    w1 = final_workloads[base_workload_alias][b]

                    first_workload_alias = args.workloads[1]
                    w2 = final_workloads[first_workload_alias][f]

                    second_workload_alias = args.workloads[2]
                    w3 = final_workloads[second_workload_alias][s]

                    w1_combined = list(zip(w1.timestamps, w1.requests))
                    w2_combined = list(zip(w2.timestamps, w2.requests))
                    w3_combined = list(zip(w3.timestamps, w3.requests))

                    merged = w1_combined + w2_combined + w3_combined
                    merged_sorted = sorted(merged, key=lambda x: x[0])
                    
                    final_timestamps, final_requests = zip(*merged_sorted)

                    modalities = [w1.modalities, w2.modalities, w3.modalities]
                    total_requests = len(w1.requests) + len(w2.requests) + len(w3.requests)
                    modality_pct = [len(w1.requests) / total_requests, len(w2.requests) / total_requests, len(w3.requests) / total_requests]

                    name = f"{base_workload_alias[0].upper()}{first_workload_alias[0].upper()}{second_workload_alias[0].upper()} with Poisson {b}-{f}-{s} v2"
                    alias = f"{base_workload_alias[0]}{first_workload_alias[0]}{second_workload_alias[0]}-poisson-{b}-{f}-{s}-v2"
                    final_workload = Workload(
                        name=name,
                        path=os.path.join(WORKLOADS_DIR, f"multi-stream"),
                        alias=alias,
                        modalities=modalities,
                        modality_pct=modality_pct,
                        modality_dist="categorical",
                        arrival_dist="poisson",
                        requests=final_requests,
                        timestamps=final_timestamps
                    )
                    final_workload.save()