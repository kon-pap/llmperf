import json
import os

from dataclasses import asdict, dataclass, field
from typing import List, LiteralString, Optional, Union

from src.constants import EXPERIMENTS_LOG, EXPERIMENTS_OUTPUTS_DIR

@dataclass
class RequestOutput:
    id: str
    prompt_tokens_cnt: int
    modality_tokens_cnt: int
    decode_tokens_cnt: int
    processor_time: float
    encoder_time: float
    ttft: float
    tbt: float
    e2e: float

    # vLLM metrics
    # TODO (konpap): Remove None after re-running static workloads
    arrival_time: float = None
    last_token_time: float = None
    first_scheduled_time: float = None
    first_token_time: float = None
    time_in_queue: float = None
    finished_time: float = None
    scheduler_time: float = None
    model_forward_time: float = None
    model_execute_time: float = None

@dataclass
class ExperimentOutput:
    id: str
    elapsed_time: float =  None
    request_outputs: List[RequestOutput] = field(default_factory=list)
    output_path: Optional[Union[str,LiteralString]] = None
    output_log_path: Optional[Union[str,LiteralString]] = None

    @property
    def num_requests(self) -> int:
        return len(self.request_outputs)
    
    @property
    def total_num_tokens(self) -> int:
        total_num_tokens = 0
        for request_output in self.request_outputs:
            prompt_tokens_cnt = request_output.prompt_tokens_cnt
            decode_tokens_cnt = request_output.decode_tokens_cnt

            total_num_tokens += prompt_tokens_cnt + decode_tokens_cnt
        return total_num_tokens
    
    @property
    def requests_per_second(self) -> float:
        return self.num_requests / self.elapsed_time
    
    @property
    def tokens_per_second(self) -> float:
        return self.total_num_tokens / self.elapsed_time


    def save(self):
        path = os.path.join(self.output_path or EXPERIMENTS_OUTPUTS_DIR, f"{self.id}.jsonl")
        with open(path, "w", encoding="utf-8") as file:
            for request_output in self.request_outputs:
                file.write(json.dumps(asdict(request_output)) + "\n")

        system_output = {
            "id": self.id,
            "elapsed_time": self.elapsed_time,
            "num_requests": self.num_requests,
            "total_num_tokens": self.total_num_tokens,
            "requests_per_second": self.requests_per_second,
            "tokens_per_second": self.tokens_per_second
        }
        with open(EXPERIMENTS_LOG, "a", encoding="utf-8") as file:
                file.write(json.dumps(system_output) + "\n")

    def load(self):
        path = os.path.join(self.output_path or EXPERIMENTS_OUTPUTS_DIR, f"{self.id}.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                self.request_outputs.append(RequestOutput(**entry))