import numpy as np
import os
import random

from llmperf.config.workloads import Workload
from llmperf.config.workloads import get_workload_by_alias
from llmperf.constants import WORKLOADS_DIR


if __name__ == '__main__':
    """
    This script creates workloads for the small benchmark
    """

    REQUEST_RATES = [1.0, 1.5, 2.0]
    SEED = 0
    TIME_QUANTUM = 10  # ms
    DURATION = 100  # seconds

    WORKLOADS = [
        "text-only",
        "long-text-light",
        "image-light",
        "video-light",
        "long-text-heavy",
        "image-heavy",
        "video-heavy",
        "mixed-light",
        "mixed-heavy"
    ]

    text_static = get_workload_by_alias("text-static")
    text_static.load()
    text_static_requests = text_static.requests
    long_text_static = get_workload_by_alias("text-static-long")
    long_text_static.load()
    long_text_static_requests = long_text_static.requests
    image_static = get_workload_by_alias("image-static")
    image_static.load()
    image_static_requests = image_static.requests
    video_static = get_workload_by_alias("video-static")
    video_static.load()
    video_static_requests = video_static.requests

    REQUEST_POOL = {
        "text": text_static_requests,
        "long-text": long_text_static_requests,
        "image": image_static_requests,
        "video": video_static_requests
    }

    random.seed(SEED)
    np.random.seed(SEED)

    final_workloads = {}
    for workload_alias in WORKLOADS:
        workload = get_workload_by_alias(workload_alias)

        for req_rate in REQUEST_RATES:
            # Generate timestamps
            lam = req_rate * (TIME_QUANTUM / 1000)
            quantums_per_sec = 1000 / TIME_QUANTUM
            arrival_times = np.random.poisson(
                lam=lam, size=int(DURATION * quantums_per_sec))
            timestamps = []
            for i, n in enumerate(arrival_times):
                timestamps += [i * (TIME_QUANTUM / 1000)] * n
            
            # Generate requests
            requests = []
            if workload_alias == "text-only":
                requests = random.choices(REQUEST_POOL["text"], k=len(timestamps))
            elif workload_alias == "long-text-light":
                # Light workload: 80% text, 20% long-text
                n_text = int(0.8 * len(timestamps))
                n_long_text = len(timestamps) - n_text
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["long-text"], k=n_long_text)
            elif workload_alias == "image-light":
                # Light workload: 80% text, 20% image
                n_text = int(0.8 * len(timestamps))
                n_image = len(timestamps) - n_text
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["image"], k=n_image)
            elif workload_alias == "video-light":
                # Light workload: 95% text, 5% video
                n_text = int(0.95 * len(timestamps))
                n_video = len(timestamps) - n_text
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["video"], k=n_video)
            elif workload_alias == "long-text-heavy":
                # Heavy workload: 50% text, 50% long-text
                n_text = int(0.5 * len(timestamps))
                n_long_text = len(timestamps) - n_text
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["long-text"], k=n_long_text)
            elif workload_alias == "image-heavy":
                # Heavy workload: 60% text, 40% image
                n_text = int(0.6 * len(timestamps))
                n_image = len(timestamps) - n_text
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["image"], k=n_image)
            elif workload_alias == "video-heavy":
                # Heavy workload: 90% text, 10% video
                n_text = int(0.9 * len(timestamps))
                n_video = len(timestamps) - n_text
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["video"], k=n_video)
            elif workload_alias == "mixed-light":
                # Mixed light workload: 60% text, 25% long-text, 10% image, 5% video
                n_text = int(0.6 * len(timestamps))
                n_long_text = int(0.25 * len(timestamps))
                n_image = int(0.1 * len(timestamps))
                n_video = len(timestamps) - n_text - n_long_text - n_image
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["long-text"], k=n_long_text) + \
                           random.choices(REQUEST_POOL["image"], k=n_image) + \
                           random.choices(REQUEST_POOL["video"], k=n_video)
            elif workload_alias == "mixed-heavy":
                # Mixed heavy workload: 40% text, 30% long-text, 20% image, 10% video
                n_text = int(0.4 * len(timestamps))
                n_long_text = int(0.3 * len(timestamps))
                n_image = int(0.2 * len(timestamps))
                n_video = len(timestamps) - n_text - n_long_text - n_image
                requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                           random.choices(REQUEST_POOL["long-text"], k=n_long_text) + \
                           random.choices(REQUEST_POOL["image"], k=n_image) + \
                           random.choices(REQUEST_POOL["video"], k=n_video)
            else:
                raise ValueError(f"Unknown workload alias: {workload_alias}")

            random.shuffle(requests)

            name = f"{workload.name} with Poisson {req_rate}"
            alias = f"{workload_alias}-{req_rate}"
            final_workload = Workload(
                name=name,
                path=os.path.join(WORKLOADS_DIR, f"small-benchmark"),
                alias=alias,
                modalities=workload.modalities,
                modality_pct=workload.modality_pct,
                arrival_dist="poisson",
                requests=requests,
                timestamps=timestamps
            )
            final_workload.save()