import hashlib
import json
import os

from dataclasses import asdict, dataclass, field
from typing import List, Literal, LiteralString, Optional, Tuple, Union

from llmperf.constants import WORKLOADS_DIR

@dataclass
class Request:
    input: str
    output: str
    id: str = None
    modality_path: Optional[Union[str,LiteralString]] = None
    # image size: (width, height) pixels
    # multi image size: [(w0, h0), (w1, h1), ...]
    # video size: duration in sec
    # audio size: duration in sec
    modality_size: Optional[Union[int,Tuple[int,int],List[Tuple[int,int]]]] = None

    def __post_init__(self): 
        if not self.id:
            combined = f"{self.input}|{self.output}|{self.modality_path or ''}"
            # ! Not collision free, but sufficient for the number of our requests
            hash_object = hashlib.sha256(combined.encode('utf-8'))
            self.id = hash_object.hexdigest()[:8]

@dataclass
class Workload:
    name: str
    path: Union[str,LiteralString]
    alias: str
    modalities: Union[str, List[str]]
    modality_pct: Union[float, List[float]]
    modality_dist: Optional[Literal["uniform"]] = None
    arrival_dist: Optional[Literal["poisson", "bursgpt", "gamma"]] = None

    requests: Optional[List[Request]] = field(default_factory=list)
    timestamps: Optional[List[float]] = field(default_factory=list)

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Workload):
            return self.name == other.name and self.alias == other.alias
        return False
    
    def save(self):
        path = os.path.join(self.path, f"{self.alias}.jsonl")
        with open(path, "w", encoding="utf-8") as file:
            for i, request in enumerate(self.requests):
                entry = {"request": asdict(request)}
                if self.timestamps:
                    entry["timestamp"] = self.timestamps[i]
                file.write(json.dumps(entry) + "\n")

    def load(self):
        path = os.path.join(self.path, f"{self.alias}.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                self.requests.append(Request(**entry["request"]))
                if "timestamp" in entry:
                    self.timestamps.append(entry["timestamp"])
        if self.timestamps:
            assert len(self.timestamps) == len(self.requests)

_WORKLOADS_STATIC = {
    Workload(
        name="Text Conversations",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="text-static",
        modalities="text",
        modality_pct=1.0
    ),
    Workload(
        name="Image Reasoning",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="image-static",
        modalities="image",
        modality_pct=1.0
    ),
    Workload(
        name="Video Description",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="video-static",
        modalities="video",
        modality_pct=1.0
    ),
    Workload(
        name="Audio Captioning",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="audio-static",
        modalities="audio",
        modality_pct=1.0
    ),
}

# Text Conversations with Poisson | Varying request rate
_WORKLOADS_TEXT_POISSON = {
    Workload(
        name="Text Conversations with Poisson 0.1",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-0.1",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 0.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-0.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 1.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-1.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 1.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-1.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 2.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-2.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 2.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-2.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 3.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-3.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 3.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-3.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 4.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-4.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 4.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-4.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 5.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-5.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 5.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-5.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 6.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-6.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 6.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-6.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 7.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-7.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 7.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-7.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 8.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-8.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 8.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-8.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 9.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-9.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 9.5",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-9.5",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Poisson 10.0",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias="text-poisson-10.0",
        arrival_dist="poisson",
        modalities="text",
        modality_pct=1.0
    )
}

# Mixed Modalities with Poisson | Varying request rate | Top 15% replaced
_WORKLOADS_MIX_POISSON_15 = {
    Workload(
        name="Mixed Modalities with Poisson 0.1 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-0.1-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 0.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-0.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 1.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-1.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 1.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-1.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 2.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-2.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 2.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-2.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 3.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-3.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 3.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-3.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 4.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-4.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 4.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-4.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 5.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-5.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 5.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-5.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 6.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-6.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 6.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-6.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 7.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-7.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 7.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-7.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 8.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-8.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 8.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-8.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 9.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-9.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 9.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-9.5-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 10.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias="mix-poisson-10.0-15",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    )
}

# Mixed Modalities with Poisson | Varying request rate | Top 30% replaced
_WORKLOADS_MIX_POISSON_30 = {
    Workload(
        name="Mixed Modalities with Poisson 0.1 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-0.1-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 0.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-0.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 1.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-1.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 1.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-1.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 2.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-2.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 2.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-2.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 3.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-3.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 3.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-3.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 4.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-4.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 4.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-4.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 5.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-5.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 5.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-5.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 6.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-6.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 6.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-6.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 7.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-7.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 7.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-7.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 8.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-8.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 8.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-8.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 9.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-9.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 9.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-9.5-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 10.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias="mix-poisson-10.0-30",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    )
}

# Mixed Modalities with Poisson | Varying request rate | Top 45% replaced
_WORKLOADS_MIX_POISSON_45 = {
    Workload(
        name="Mixed Modalities with Poisson 0.1 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-0.1-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 0.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-0.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 1.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-1.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 1.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-1.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 2.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-2.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 2.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-2.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 3.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-3.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 3.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-3.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 4.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-4.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 4.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-4.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 5.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-5.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 5.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-5.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 6.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-6.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 6.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-6.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 7.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-7.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 7.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-7.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 8.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-8.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 8.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-8.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 9.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-9.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 9.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-9.5-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Poisson 10.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias="mix-poisson-10.0-45",
        arrival_dist="poisson",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    )
}

# Text Conversations with Gamma | Varying request rate
_WORKLOADS_TEXT_GAMMA = {
    Workload(
        name="Text Conversations with Gamma 0.1",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-0.1",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 0.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-0.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 1.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-1.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 1.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-1.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 2.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-2.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 2.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-2.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 3.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-3.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 3.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-3.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 4.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-4.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 4.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-4.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 5.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-5.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 5.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-5.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 6.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-6.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 6.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-6.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 7.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-7.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 7.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-7.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 8.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-8.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 8.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-8.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 9.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-9.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 9.5",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-9.5",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    ),

    Workload(
        name="Text Conversations with Gamma 10.0",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias="text-gamma-10.0",
        arrival_dist="gamma",
        modalities="text",
        modality_pct=1.0
    )
}

# Mixed Modalities with Gamma | Varying request rate | Top 15% replaced
_WORKLOADS_MIX_GAMMA_15 = {
    Workload(
        name="Mixed Modalities with Gamma 0.1 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-0.1-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 0.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-0.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 1.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-1.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 1.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-1.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 2.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-2.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 2.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-2.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 3.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-3.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 3.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-3.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 4.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-4.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 4.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-4.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 5.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-5.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 5.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-5.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 6.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-6.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 6.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-6.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 7.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-7.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 7.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-7.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 8.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-8.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 8.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-8.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 9.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-9.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 9.5 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-9.5-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 10.0 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias="mix-gamma-10.0-15",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.85, 0.05, 0.05, 0.05],
        modality_dist="uniform"
    )
}

# Mixed Modalities with Gamma | Varying request rate | Top 30% replaced
_WORKLOADS_MIX_GAMMA_30 = {
    Workload(
        name="Mixed Modalities with Gamma 0.1 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-0.1-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 0.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-0.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 1.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-1.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 1.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-1.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 2.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-2.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 2.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-2.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 3.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-3.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 3.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-3.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 4.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-4.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 4.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-4.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 5.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-5.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 5.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-5.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 6.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-6.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 6.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-6.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 7.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-7.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 7.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-7.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 8.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-8.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 8.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-8.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 9.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-9.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 9.5 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-9.5-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 10.0 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias="mix-gamma-10.0-30",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.7, 0.1, 0.1, 0.1],
        modality_dist="uniform"
    )
}

# Mixed Modalities with Gamma | Varying request rate | Top 45% replaced
_WORKLOADS_MIX_GAMMA_45 = {
        Workload(
        name="Mixed Modalities with Gamma 0.1 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-0.1-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 0.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-0.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 1.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-1.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 1.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-1.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 2.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-2.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 2.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-2.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 3.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-3.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 3.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-3.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 4.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-4.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 4.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-4.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 5.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-5.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 5.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-5.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 6.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-6.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 6.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-6.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 7.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-7.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 7.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-7.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 8.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-8.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 8.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-8.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 9.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-9.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 9.5 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-9.5-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    ),
    Workload(
        name="Mixed Modalities with Gamma 10.0 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias="mix-gamma-10.0-45",
        arrival_dist="gamma",
        modalities=["text", "image", "video", "audio"],
        modality_pct=[0.55, 0.15, 0.15, 0.15],
        modality_dist="uniform"
    )
}

WORKLOADS = _WORKLOADS_STATIC | \
    _WORKLOADS_TEXT_POISSON | \
    _WORKLOADS_MIX_POISSON_15 | \
    _WORKLOADS_MIX_POISSON_30 | \
    _WORKLOADS_MIX_POISSON_45 | \
    _WORKLOADS_TEXT_GAMMA | \
    _WORKLOADS_MIX_GAMMA_15 | \
    _WORKLOADS_MIX_GAMMA_30 | \
    _WORKLOADS_MIX_GAMMA_45

def get_workload_by_name(name: str) -> Union[None, Workload]:
    for workload in WORKLOADS:
        if getattr(workload, "name", None) == name:
            return workload
    return None

def get_workload_by_alias(alias: str) -> Union[None, Workload]:
    for workload in WORKLOADS:
        if getattr(workload, "alias", None) == alias:
            return workload
    return None