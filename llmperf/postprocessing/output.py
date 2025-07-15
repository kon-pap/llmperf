import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataclasses import asdict, dataclass, field
from typing import List, LiteralString, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import RequestOutput as vllmRequestOutput

from llmperf.constants import EXPERIMENTS_LOG, EXPERIMENTS_OUTPUTS_DIR, EXPERIMENTS_ENGINE_STATS_DIR

@dataclass
class RequestOutput:
    id: str
    prompt_tokens_cnt: Optional[int] = 0
    modality_tokens_cnt: Optional[int] = 0
    decode_tokens_cnt: Optional[int] = 0

    # vLLM metrics
    vllm_id: Optional[int] = 0
    arrival_time: Optional[float] = 0.0
    input_processed_time: Optional[float] = 0.0
    last_token_time: Optional[float] = 0.0
    first_scheduled_time: Optional[float] = 0.0
    first_token_time: Optional[float] = 0.0
    time_in_queue: Optional[float] = 0.0
    finished_time: Optional[float] = 0.0
    scheduler_time: Optional[float] = 0.0
    model_forward_time: Optional[float] = 0.0
    model_execute_time: Optional[float] = 0.0
    model_encoder_time: Optional[float] = 0.0

    # Metadata
    estimated_time: Optional[float] = None
    category: Optional[str] = None
    stl: Optional[bool] = None
    aborted: Optional[bool] = None
    uid: Optional[str] = None
    slo: Optional[float] = None
    ttft_slo: Optional[float] = None
    tbt_slo: Optional[float] = None

    @property
    def processor_time(self) -> float:
        return self.input_processed_time - self.arrival_time
    
    @property
    def encoder_time(self) -> float:
        return self.model_encoder_time
    
    @property
    def ttft(self) -> float:
        return self.processor_time + self.time_in_queue + (self.first_token_time - self.first_scheduled_time)
    
    @property
    def tbt(self) -> float:
        return 0.0 if self.decode_tokens_cnt <= 1 else (self.e2e - self.ttft) / (self.decode_tokens_cnt - 1)
    
    @property
    def e2e(self) -> float:
        return self.finished_time - self.arrival_time
    
    @property
    def queued_time(self) -> float:
        return self.time_in_queue
    
    @property
    def prefill_time(self) -> float:
        return self.first_token_time - self.first_scheduled_time
    
    @property
    def decode_time(self) -> float:
        return self.last_token_time - self.first_token_time
    
    @property
    def inference_time(self) -> float:
        return self.last_token_time - self.first_scheduled_time
    
    @classmethod
    def from_vllm_output(cls, req_id: str, req_output: "vllmRequestOutput", modality_token_index: int = -1) -> "RequestOutput":
        aborted = (
            len(req_output.outputs[0].token_ids) == 1 and \
            req_output.outputs[0].token_ids[0] == -1
        )

        estimated_time = category = stl = None
        if md := req_output.request_md:
            estimated_time = md.estimated_time
            category = md.category
            stl = md.stl

        uid = slo = ttft_slo = tbt_slo = None

        if (hasattr(req_output, "user_md")) and (umd := req_output.user_md):
            uid = umd.uid
            slo = umd.slo
            ttft_slo = umd.ttft_slo
            tbt_slo = umd.tbt_slo

        return cls(
            id=req_id,
            prompt_tokens_cnt=len(req_output.prompt_token_ids),
            modality_tokens_cnt=req_output.prompt_token_ids.count(modality_token_index),
            decode_tokens_cnt=len(req_output.outputs[0].token_ids),
            vllm_id=req_output.request_id,
            arrival_time=req_output.metrics.arrival_time,
            input_processed_time=req_output.metrics.input_processed_time,
            last_token_time=req_output.metrics.last_token_time,
            first_scheduled_time=req_output.metrics.first_scheduled_time,
            first_token_time=req_output.metrics.first_token_time,
            time_in_queue=req_output.metrics.time_in_queue,
            finished_time=req_output.metrics.finished_time,
            scheduler_time=req_output.metrics.scheduler_time,
            model_forward_time=req_output.metrics.model_forward_time,
            model_execute_time=req_output.metrics.model_execute_time,
            model_encoder_time=req_output.metrics.model_encoder_time,
            estimated_time=estimated_time,
            category=category,
            stl=stl,
            aborted=aborted,
            uid=uid,
            slo=slo,
            ttft_slo=ttft_slo,
            tbt_slo=tbt_slo
        )

@dataclass
class EngineStats:
    timestamps: List[float]
    kv_cache_usage: List[float]
    preemptions_req_ids: List[List[str]]

    num_preemptions: List[int] = field(init=False)

    def __post_init__(self):
        self.num_preemptions = [len(reqs) for reqs in self.preemptions_req_ids]

@dataclass
class ExperimentOutput:
    id: str
    elapsed_time: float =  None
    request_outputs: List[RequestOutput] = field(default_factory=list)
    output_path: Optional[Union[str,LiteralString]] = None
    output_log_path: Optional[Union[str,LiteralString]] = None
    engine_stats_path: Optional[Union[str,LiteralString]] = None
    engine_stats: Optional[EngineStats] = None

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


    def save_engine_stats(self, log_file: Union[str,LiteralString]):
        path = os.path.join(self.engine_stats_path or EXPERIMENTS_ENGINE_STATS_DIR, f"{self.id}.parquet")

        df = pd.read_json(log_file, lines=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression="brotli", compression_level=11)

    def load_engine_stats(self):
        path = os.path.join(self.engine_stats_path or EXPERIMENTS_ENGINE_STATS_DIR, f"{self.id}.parquet")
        df = pd.read_parquet(path)
        
        self.engine_stats = EngineStats(
            timestamps=df["timestamp"].tolist(),
            kv_cache_usage=df["kv_cache_usage"].tolist(),
            preemptions_req_ids=[list(arr) for arr in df["preempted_req_ids"].tolist()]
        )

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

        if self.elapsed_time is None:
            start_time = min([ro.arrival_time for ro in self.request_outputs])
            finish_time = max([ro.finished_time for ro in self.request_outputs])
            self.elapsed_time = finish_time - start_time