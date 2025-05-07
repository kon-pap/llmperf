import time

from datetime import datetime
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.video import VideoAsset

from llmperf.config.approaches import get_approach_by_name
from llmperf.config.models import get_model_by_name
from llmperf.config.workloads import get_workload_by_name
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import load_image

if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
    FRAMES = 64

    GPU_UTIL = 0.95
    SWAP_SPACE = 0
    approach = get_approach_by_name("Isolation")
    
    workload_names = [
        "Text Conversations",
        "Image Reasoning",
        "Video Description",
        "Audio Captioning"
    ]

    model_names = [
        "Mistral-7b",
        "LLaVA 1.6 (Mistral-7b)",
        "LLaVA-Next-Video (Mistral-7b)",
        "Qwen2-Audio-7b"
    ]

    for workload_name, model_name in zip(workload_names, model_names):
        try:
            model = get_model_by_name(model_name)

            llm = LLM(
                model=model.path,
                gpu_memory_utilization=GPU_UTIL,
                swap_space=SWAP_SPACE,
                scheduling_policy=approach.scheduling_policy,
                enable_custom_scheduler=approach.enable_custom_scheduler,
                enable_chunked_prefill=approach.enable_chunked_prefill
            )

            workload = get_workload_by_name(workload_name)
            workload.load()
            requests = workload.requests

            outputs = []
            modality_token_index = -1
            start_time = time.time()
            for request in tqdm(requests):
                tokenizer = AutoTokenizer.from_pretrained(model.path)
                output_length = len(tokenizer.encode(request.output))

                sampling_params = SamplingParams(
                    ignore_eos=True,
                    max_tokens=int(output_length)
                )

                if workload.alias == "text-static":
                    modality_token_index = -1
                    tokenizer = AutoTokenizer.from_pretrained(model.path)
                    prompt = [
                        {"role": "user", "content": request.input}
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                    final_prompt = {
                        "prompt": formatted_prompt
                    }

                if workload.alias == "image-static":
                    modality_token_index = model.image_token_index
                    processor = AutoProcessor.from_pretrained(model.path)
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": request.input}
                            ]
                        }
                    ]
                    formatted_prompt = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    
                    image = load_image(request.modality_path)

                    final_prompt = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"image": image}
                    }

                if workload.alias == "video-static":
                    modality_token_index = model.video_token_index
                    processor = AutoProcessor.from_pretrained(model.path)
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video"},
                                {"type": "text", "text": request.input}
                            ]
                        }
                    ]
                    formatted_prompt = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                    video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays

                    final_prompt = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"video": video}
                    }

                if workload.alias == "audio-static":
                    modality_token_index = model.audio_token_index
                    processor = AutoProcessor.from_pretrained(model.path)

                    audio_in_prompt = "".join([
                        f"Audio 1: "
                        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                    ])
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio"},
                                {"type": "text", "text": audio_in_prompt + request.input}
                            ]
                        }
                    ]
                    formatted_prompt = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                    audio = AudioAsset(request.modality_path).audio_and_sample_rate

                    final_prompt = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"audio": audio}
                    }

                req_output = llm.generate(
                    prompts=[final_prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )[0]

                now = time.time()
                outputs.append(
                    RequestOutput.from_vllm_output(
                        request.id, req_output, modality_token_index
                    )
                )

        finally:
            elapsed_time = now - start_time

            experiment_output = ExperimentOutput(
                id=f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}",
                elapsed_time=elapsed_time,
                request_outputs=outputs
            )
            experiment_output.save()

            llm = None