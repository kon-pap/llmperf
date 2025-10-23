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
from llmperf.postprocessing.aggregator import Aggregator
from llmperf.postprocessing.filter import Filter

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
    encoder_cache_usage: List[float]
    preempted_req_ids: List[List[str]]
    preempted_req_ts: List[List[float]]
    rescheduled_req_ids: List[List[str]]
    rescheduled_req_ts: List[List[float]]
    kv_cache_usage_per_category: List[dict[str,float]]
    encoder_cache_usage_per_category: List[dict[str,float]]

    num_preemptions: List[int] = field(init=False)

    def __post_init__(self):
        self.num_preemptions = [len(reqs) for reqs in self.preempted_req_ids]

@dataclass
class ExperimentOutput:
    id: str
    elapsed_time: float =  None
    request_outputs: List[RequestOutput] = field(default_factory=list)
    output_path: Optional[Union[str,LiteralString]] = None
    output_log_path: Optional[Union[str,LiteralString]] = None
    engine_stats_path: Optional[Union[str,LiteralString]] = None
    engine_stats: Optional[EngineStats] = None

    LATENCY_SLO_MAP = {
        "e2e": ("e2e", "slo"),
        "tbt": ("tbt", "tbt_slo"),
        "ttft": ("ttft", "ttft_slo"),
    }

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

    def _latency(self, type: str, method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        request_latencies = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            if type == "normalized":
                request_latencies.append(ro.e2e / ro.decode_tokens_cnt)
            else:
                request_latencies.append(getattr(ro, type))
        
        return Aggregator.aggregate(request_latencies, method)

    def normalized_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("normalized", method, filter)

    def ttft_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("ttft", method, filter)

    def e2e_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("e2e", method, filter)

    def tbt_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("tbt", method, filter)
    
    def tiq_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("time_in_queue", method, filter)
    
    def preemption_latency(self, method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        preemption_deltas = {ro.id: [] for ro in self.request_outputs if filter.include(ro)}
        req_id_to_ro = { ro.id: ro for ro in self.request_outputs }

        active_preemptions = {}

        timestamps = self.engine_stats.timestamps
        preempted_lists = self.engine_stats.preemptions_req_ids
        rescheduled_lists = self.engine_stats.rescheduled_req_ids

        for ts, preempted, rescheduled in zip(timestamps, preempted_lists, rescheduled_lists):
            # Handle new preemptions (start of preemption)
            for req_id in preempted:
                if req_id in preemption_deltas and req_id not in active_preemptions:
                    active_preemptions[req_id] = ts

            # Handle reschedules (end of preemption)
            for req_id in rescheduled:
                if req_id in active_preemptions:
                    start = active_preemptions.pop(req_id)
                    delta = ts - start
                    preemption_deltas[req_id].append(delta)
                else:
                    print("Something went wrong")

        assert not active_preemptions

        preemption_latencies = []
        for req_id, deltas in preemption_deltas.values():
            ro = req_id_to_ro[req_id]
            if not filter.include(ro):
                continue
            preemption_latencies.append(sum(deltas))
        
        return Aggregator.aggregate(preemption_latencies, method)
    
    def latency_slo_delta(self, latency_type: str = "e2e", method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        request_deltas = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
             # latency - slo thres
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)
            request_deltas.append(latency - slo)

        return Aggregator.aggregate(request_deltas, method)

    def slo_headroom(self, latency_type: str = "e2e", method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        request_headrooms = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)

            # slo thres - latency (for negative deltas = for positive headrooms)
            headroom = slo - latency
            if headroom > 0:
                request_headrooms.append(headroom)

        return Aggregator.aggregate(request_headrooms, method)

    def slo_violation(self, latency_type: str = "e2e", method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        request_violations = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)

            # latency - slo thres (for positive deltas = for negative headrooms)
            violation = latency - slo
            if violation > 0:
                request_violations.append(violation)

        return Aggregator.aggregate(request_violations, method)

    def slo_violations(self, latency_type: str = "e2e", filter: Filter = None) -> int:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        num_violations = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)
            if latency > slo:
                num_violations += 1

        return num_violations

    def _slo_attainment(self, latency_attr: str, slo_attr: str, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        if slo_map is None:
            slo_map = {}

        filtered_cnt = 0
        attained_cnt = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue

            filtered_cnt += 1

            slo = getattr(ro, slo_attr)
            if slo is None or slo == 0.0:
                slo = slo_map.get(ro.id, float("inf"))

            if getattr(ro, latency_attr) <= slo:
                attained_cnt += 1

        return attained_cnt / filtered_cnt * 100 if filtered_cnt else 0.0

    def e2e_slo_attainment(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        return self._slo_attainment("e2e", "slo", filter, slo_map)

    def tbt_slo_attainment(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        return self._slo_attainment("tbt", "tbt_slo", filter, slo_map)

    def ttft_slo_attainment(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        return self._slo_attainment("ttft", "ttft_slo", filter, slo_map)
    
    def throughput(self, filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        request_cnt = 0
        start_time = float("inf")
        finished_time = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            start_time = min(start_time, ro.arrival_time)
            finished_time = max(finished_time, ro.finished_time)
            request_cnt += 1
        
        return (finished_time - start_time) / request_cnt if request_cnt else 0
    
    def goodput(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        if slo_map is None:
            slo_map = {}

        attained_cnt = 0
        start_time = float("inf")
        finished_time = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue

            slo = ro.slo
            if slo is None:
                slo = slo_map.get(ro.id, float("inf"))

            # TODO: Rethink goodput SLO definition (TTFT?, TBT?)
            if ro.e2e <= slo:
                attained_cnt += 1

            start_time = min(start_time, ro.arrival_time)
            finished_time = max(finished_time, ro.finished_time)

        return (finished_time - start_time) / attained_cnt if attained_cnt else 0

    def preemptions(self, filter: Filter = None) -> int:
        if filter is None:
            return sum(self.engine_stats.num_preemptions)
        
        req_id_to_ro = { ro.vllm_id: ro for ro in self.request_outputs }

        num_preemptions = 0
        for preempted_req_ids in self.engine_stats.preempted_req_ids:
            for req_id in preempted_req_ids:
               if not filter.include(req_id_to_ro[req_id]):
                   continue
               num_preemptions += 1
               
        return num_preemptions
    
    def aborted(self, filter: Filter = None) -> int:
        if filter is None:
            filter = Filter()
        
        aborted = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            if ro.aborted:
                aborted += 1
               
        return aborted

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
            encoder_cache_usage=df["encoder_cache_usage"].tolist(),
            rescheduled_req_ids=[list(arr) for arr in df["rescheduled_req_ids"].tolist()],
            rescheduled_req_ts=[list(arr) for arr in df["rescheduled_req_ts"].tolist()],
            preempted_req_ids=[list(arr) for arr in df["preempted_req_ids"].tolist()],
            preempted_req_ts=[list(arr) for arr in df["preempted_req_ts"].tolist()],
            kv_cache_usage_per_category=df["kv_cache_usage_per_category"].tolist(),
            encoder_cache_usage_per_category=df["encoder_cache_usage_per_category"].tolist()
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