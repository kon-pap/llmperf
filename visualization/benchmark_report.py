import argparse
import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import time
import json
from collections import defaultdict
from functools import partial

plt.style.use('fivethirtyeight')
from llmperf.postprocessing.output import ExperimentOutput
from llmperf.postprocessing.filter import Filter
from llmperf.constants import FIGURES_DIR, ARTIFACTS_DIR

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

WORKLOAD_MAP = {
    "text-only": "to",
    "long-text-light": "ltl",
    "long-text-heavy": "lth",
    "image-light": "il",
    "image-heavy": "ih",
    "video-light": "vl",
    "video-heavy": "vh",
    "mixed-light": "ml",
    "mixed-heavy": "mh",
}

WORKLOAD_NAME_MAP = {
    "to": "Text-Only",
    "ltl": "Long Text Light",
    "lth": "Long Text Heavy",
    "il": "Image Light",
    "ih": "Image Heavy",
    "vl": "Video Light",
    "vh": "Video Heavy",
    "ml": "Mixed Light",
    "mh": "Mixed Heavy",
}

BASELINE_MAP = {
    "v0.0.1.baseline": "vllm",
    "v0.0.1.naive-classifier": "naive-classifier",
    "v0.0.1.naive-slo-aware": "naive-e2e-slo-aware",
    "v0.0.2.naive-slo-aware": "naive-ttft-slo-aware",
    "v0.0.1.naive-mlq": "naive-mlq",
    "wip": "mlq-lkv",
    "wip-ii": "mlq-sof",
    "wip-iii": "mlq-mkv"
}

BASELINE_NAME_MAP = {
    "vllm": "vLLM",
    "naive-classifier": "Naive Classifier",
    "naive-e2e-slo-aware": "Naive E2E SLO-Aware",
    "naive-ttft-slo-aware": "Naive TTFT SLO-Aware",
    "naive-mlq": "Naive MLQ",
    "mlq-lkv": "MLQ-LKV",
    "mlq-sof": "MLQ-SOF",
    "mlq-mkv": "MLQ-MKV"
}

MEMORY_MAP = {
    "blocksdef": "max",
    "blocks512": "512",
    "blocks256": "256"
}

MEMORY_NAME_MAP = {
    "max": "Max Memory",
    "512": "512 Blocks",
    "256": "256 Blocks"
}

COLOR_MAP = {
    "vllm": "#008fd5",
    "naive-classifier": "#fc4f30",
    "naive-e2e-slo-aware": "#e5ae38",
    "naive-ttft-slo-aware": "#6d904f",
    "naive-mlq": "#810f7c",
    "mlq-lkv": "#8b8b8b",
    "mlq-sof": "#810f7c",
    "mlq-mkv": "#000000"
}

ID_MAP = {
    "sand": set(),
    "pebbles": set(),
    "rocks": set(),
    "texts": set(),
    "long-texts": set(),
    "images": set(),
    "videos": set()
}

T_ID = "text-static-llava-ov-iso-20250514-192704"
I_ID = "image-static-llava-ov-iso-20250514-204328"
V_ID = "video-static-llava-ov-iso-20250515-235113"
LT_ID = "text-static-long__llava-ov__iso__v0.0.1.naive-slo-aware__20250910-165223__maxlendef__batchdef__blocksdef__gpu0.95__swap0"

E2E_SLO_SCALE = 5
TBT_SLO_SCALE = 5
TTFT_SLO_SCALE = 6

def plot_single(workloads, baselines, values, colors, ylabel, title, filename):
    x = np.arange(len(workloads))
    width = 0.8 / len(baselines)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw bars for each baseline
    for i, baseline in enumerate(baselines):
        ax.bar(x + i*width, values[:, i], width, label=baseline, color=colors[i])

    # Formatting
    ax.set_xlabel("Workloads")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax.set_xticklabels(workloads)

    fig.tight_layout()
    file_path = os.path.join(FIGURES_DIR, "benchmark", f"{filename}.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.close()

def plot_helper(experiment_outputs, workload_names, ylabel, title_prefix, file_prefix, compute_fn_name, **kwargs):
    for memory in experiment_outputs.keys():
        baseline_names = [BASELINE_NAME_MAP[b] for b in experiment_outputs[memory].keys()]
        colors = [COLOR_MAP[b] for b in experiment_outputs[memory].keys()]
        metrics = []
        for baseline in experiment_outputs[memory].keys():
            workload_metrics = []
            for workload in experiment_outputs[memory][baseline]:
                if experiment_outputs[memory][baseline][workload] is None:
                    workload_metrics.append(-10**(-20))
                    continue
                
                eo = experiment_outputs[memory][baseline][workload]
                metric = getattr(eo, compute_fn_name)(**kwargs)
                
                workload_metrics.append(metric)
            
            metrics.append(workload_metrics)

        title = f"{title_prefix} | {MEMORY_NAME_MAP[memory]}"
        filename = f"{file_prefix}_{memory}"
        plot_single(workload_names, baseline_names, np.array(metrics).T, colors, ylabel, title, filename)

def plot_single_report(input_paths, out_path, rows, cols):
    W, H = 1200, 540
    cell_w, cell_h = W / cols, H / rows

    out_doc = fitz.open()
    total = len(input_paths)

    idx = 0
    while idx < total:
        page = out_doc.new_page(width=W, height=H)
        for cell_index in range(rows * cols):
            if idx >= total:
                break
            src_path = str(input_paths[idx])
            try:
                src_doc = fitz.open(src_path)
            except Exception as e:
                print(f"❌ Cannot open '{src_path}': {e}")
                idx += 1
                continue

            if src_doc.page_count < 1:
                print(f"⚠️ '{src_path}' has no pages")
                src_doc.close()
                idx += 1
                continue

            col = cell_index % cols
            row = cell_index // cols
            x0 = col * cell_w
            y0 = row * cell_h
            rect = fitz.Rect(x0, y0, x0 + cell_w, y0 + cell_h)

            try:
                page.show_pdf_page(rect, src_doc, 0)
            except Exception as e:
                print(f"❌ Failed to place '{src_path}' in the report: {e}")
            finally:
                src_doc.close()
            idx += 1

    out_doc.save(out_path)
    out_doc.close()
    
def plot_reports(memory_configs):
    # Normalize Latency Reports
    for memory in memory_configs:
        prefixes = ["", "p50", "p90", "p99"]
        categories = ["", "sand", "pebbles", "rocks"]

        image_files = [
            f"normlat_{'_'.join(filter(None, [cat, suf]))}_{memory}.pdf".replace("__", "_")
            for cat in categories
            for suf in prefixes
        ]

        image_paths = [os.path.join(FIGURES_DIR, "benchmark", file) for file in image_files]
        report_path = os.path.join(FIGURES_DIR, "reports", f"normlat_{memory}.pdf")
        plot_single_report(image_paths, report_path, 4, 4)

    # TTFT Latency Reports
    for memory in memory_configs:
        prefixes = ["", "p50", "p90", "p99"]
        categories = ["", "sand", "pebbles", "rocks"]

        image_files = [
            f"ttft_{'_'.join(filter(None, [cat, suf]))}_{memory}.pdf".replace("__", "_")
            for cat in categories
            for suf in prefixes
        ]

        image_paths = [os.path.join(FIGURES_DIR, "benchmark", file) for file in image_files]
        report_path = os.path.join(FIGURES_DIR, "reports", f"ttft_{memory}.pdf")
        plot_single_report(image_paths, report_path, 4, 4)

    # Throughput Reports
    for memory in memory_configs:
        categories = ["", "sand", "pebbles", "rocks"]

        image_files = [f"tp_{cat}_{memory}.pdf".replace("__", "_") for cat in categories]

        image_paths = [os.path.join(FIGURES_DIR, "benchmark", file) for file in image_files]
        report_path = os.path.join(FIGURES_DIR, "reports", f"tp_{memory}.pdf")
        plot_single_report(image_paths, report_path, 1, 4)

    # E2E SLO | TTFT SLO | Aborted | Preemptions Reports
    for memory in memory_configs:
        metrics = ["e2e_slo", "ttft_slo", "aborted", "preemptions"]
        categories = ["", "sand", "pebbles", "rocks"]

        image_files = [
            f"{metric}{'_' + cat if cat else ''}_{memory}.pdf"
            for cat in categories
            for metric in metrics
        ]

        image_paths = [os.path.join(FIGURES_DIR, "benchmark", file) for file in image_files]
        report_path = os.path.join(FIGURES_DIR, "reports", f"slo_{memory}.pdf")
        plot_single_report(image_paths, report_path, 4, 4)

def plot_legend():
    legend_handles = [
        Patch(facecolor=COLOR_MAP[key], label=BASELINE_NAME_MAP[key])
        for key in BASELINE_NAME_MAP
    ]

    fig, ax = plt.subplots()
    ax.legend(handles=legend_handles, title="Baselines", loc="center", ncols=len(legend_handles))
    ax.axis("off")

    file_path = os.path.join(FIGURES_DIR, "benchmark", f"legend.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.close()

def parse_workloads(aliases):
    workload_names = []
    for workload_alias in aliases:
        if workload_alias in WORKLOAD_NAME_MAP:
            workload_names.append(WORKLOAD_NAME_MAP[workload_alias])
        else:
            raise ValueError(f"Unknown workload alias: {workload_alias}")
    return workload_names

def parse_logs(filepath: str):
    experiment_ids = defaultdict(lambda: defaultdict(dict))

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")
                continue

            params = record["id"].split("__")
            workload_name = params[0]

            if params[3] not in BASELINE_MAP:
                continue

            baseline = BASELINE_MAP[params[3]]
            memory = MEMORY_MAP[params[7]]

            for prefix, alias in WORKLOAD_MAP.items():
                if workload_name.startswith(prefix):
                    workload = alias
                    break
            else:
                raise ValueError(f"Unknown workload alias: {workload_name}")

            experiment_ids[memory][baseline][workload] = record["id"]

    return experiment_ids

def load_experiment_outputs(experiment_ids, workloads):
    experiment_outputs = defaultdict(lambda: defaultdict(dict))
    for memory in experiment_ids.keys():
        for baseline in experiment_ids[memory].keys():
            for workload in workloads:
                if workload not in experiment_ids[memory][baseline].keys():
                    experiment_outputs[memory][baseline][workload] = None
                    continue
                
                eo_id = experiment_ids[memory][baseline][workload]

                if "wip-iii" in eo_id:
                    eo_id = eo_id.replace("wip-iii", "v0.0.2.naive-slo-aware")
                
                if "wip-ii" in eo_id:
                    eo_id = eo_id.replace("wip-ii", "v0.0.2.naive-slo-aware")

                if "wip" in eo_id:
                    eo_id = eo_id.replace("wip", "v0.0.2.naive-slo-aware")
                
                eo = ExperimentOutput(id=eo_id)
                eo.load()
                eo.load_engine_stats()
                experiment_outputs[memory][baseline][workload] = eo
    return experiment_outputs

def load_id_map():
    t_eo = ExperimentOutput(id=T_ID)
    t_eo.load()
    i_eo = ExperimentOutput(id=I_ID)
    i_eo.load()
    v_eo = ExperimentOutput(id=V_ID)
    v_eo.load()
    lt_eo = ExperimentOutput(id=LT_ID)
    lt_eo.load()

    ID_MAP["sand"] = set([ro.id for ro in t_eo.request_outputs] + [ro.id for ro in lt_eo.request_outputs])
    ID_MAP["pebbles"] = ID_MAP["images"] = set([ro.id for ro in i_eo.request_outputs])
    ID_MAP["rocks"] = ID_MAP["videos"] = set([ro.id for ro in v_eo.request_outputs])
    ID_MAP["texts"] = set([ro.id for ro in t_eo.request_outputs])
    ID_MAP["long-texts"] = set([ro.id for ro in lt_eo.request_outputs])

def load_slo_map(slo_type):
    t_eo = ExperimentOutput(id=T_ID)
    t_eo.load()
    i_eo = ExperimentOutput(id=I_ID)
    i_eo.load()
    v_eo = ExperimentOutput(id=V_ID)
    v_eo.load()
    lt_eo = ExperimentOutput(id=LT_ID)
    lt_eo.load()
    
    request_outputs = t_eo.request_outputs + i_eo.request_outputs + v_eo.request_outputs + lt_eo.request_outputs
    
    id_to_slo = {}
    for ro in request_outputs:
        if slo_type == "e2e":
            id_to_slo[ro.id] = ro.e2e * E2E_SLO_SCALE
        elif slo_type == "ttft":
            id_to_slo[ro.id] = ro.ttft * TTFT_SLO_SCALE
        elif slo_type == "tbt":
            id_to_slo[ro.id] = ro.tbt * TBT_SLO_SCALE
        else:
            raise ValueError(f"Unsupported SLO type: {slo_type}")

    return id_to_slo

if __name__ == "__main__":
    total_start = time.time()

    benchmark_logs = os.path.join(ARTIFACTS_DIR, "benchmark-log.jsonl")

    parser = argparse.ArgumentParser(description="Plot grouped bar chart for workloads and baselines.")
    parser.add_argument("workloads", nargs="+", help="List of workload names")
    parser.add_argument("--reports", action="store_true", help="Print the powerpoint reports")
    args = parser.parse_args()

    workload_names = parse_workloads(args.workloads)
    experiment_ids = parse_logs(benchmark_logs)
    experiment_outputs = load_experiment_outputs(experiment_ids, args.workloads)
    
    load_id_map()

    sand_filter = Filter(ID_MAP["sand"], "sand")
    pebbles_filter = Filter(ID_MAP["pebbles"], "pebbles")
    rocks_filter = Filter(ID_MAP["rocks"], "rocks")
    text_filter = Filter(ID_MAP["texts"])
    long_text_filter = Filter(ID_MAP["long-texts"])
    image_filter = Filter(ID_MAP["images"])
    video_filter = Filter(ID_MAP["videos"])

    e2e_slo_map = load_slo_map("e2e")
    ttft_slo_map = load_slo_map("ttft")
    tbt_slo_map = load_slo_map("tbt")

    # Plots
    plot = partial(
        plot_helper,
        experiment_outputs,
        workload_names
    )
    
    # Common request types
    request_types = [
        ("All Requests", "all", None),
        ("Sand Requests", "sand", sand_filter),
        ("Pebble Requests", "pebbles", pebbles_filter),
        ("Rock Requests", "rocks", rocks_filter),
        ("Text Requests", "text", text_filter),
        ("Image Requests", "image", image_filter),
        ("Video Requests", "video", video_filter),
        ("Long Text Requests", "long_text", long_text_filter),
    ]

    # Percentile-based metrics (support p99, p90, p50 variants)
    percentile_methods = [
        (None, ""),
        ("p99", " | P99"),
        ("p90", " | P90"),
        ("p50", " | P50"),
    ]

    # Define all metrics in one place
    metrics = [
        {
            "ylabel": "Norm. Lat. (s/tkn)",
            "file_prefix": "normlat",
            "compute_fn": "normalized_latency",
            "supports_percentiles": True,
        },
        {
            "ylabel": "TTFT (s)",
            "file_prefix": "ttft",
            "compute_fn": "ttft_latency",
            "supports_percentiles": True,
        },
        {
            "ylabel": "Preemption Latency (s)",
            "file_prefix": "prelat",
            "compute_fn": "preemption_latency",
            "supports_percentiles": True,
        },
        {
            "ylabel": "Throughput (req/s)",
            "file_prefix": "tp",
            "compute_fn": "throughput",
            "supports_percentiles": False,
        },
        {
            "ylabel": "E2E SLO Attainment (%)",
            "file_prefix": "e2e_slo",
            "compute_fn": "e2e_slo_attainment",
            "extra_args": {"slo_map": e2e_slo_map},
            "supports_percentiles": False,
        },
        {
            "ylabel": "TTFT SLO Attainment (%)",
            "file_prefix": "ttft_slo",
            "compute_fn": "ttft_slo_attainment",
            "extra_args": {"slo_map": ttft_slo_map},
            "supports_percentiles": False,
        },
        {
            "ylabel": "Num. of Preemptions",
            "file_prefix": "preemptions",
            "compute_fn": "preemptions",
            "supports_percentiles": False,
        },
        {
            "ylabel": "Num. of Aborted",
            "file_prefix": "aborted",
            "compute_fn": "aborted",
            "supports_percentiles": False,
        },
    ]

    for metric in metrics:
        for title, prefix, fltr in request_types:
            methods = percentile_methods if metric.get("supports_percentiles") else [(None, "")]
            for method, suffix in methods:
                kwargs = {
                    "ylabel": metric["ylabel"],
                    "title_prefix": f"{title}{suffix}",
                    "file_prefix": f"{metric['file_prefix']}{'_' + prefix if prefix != 'all' else ''}{'_' + method if method else ''}",
                    "compute_fn_name": metric["compute_fn"],
                }

                if "extra_args" in metric:
                    kwargs.update(metric["extra_args"])

                if method:
                    kwargs["method"] = method
                if fltr:
                    kwargs["filter"] = fltr

                plot(**kwargs)

    if args.reports:
        plot_reports(memory_configs=experiment_outputs.keys())

    plot_legend()

    total_end = time.time()
    total_duration = total_end - total_start
    print(f"=== Total execution time: {total_duration:.2f} seconds ===")