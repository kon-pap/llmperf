import ast

from typing import Dict, Optional

def get_sharegpt_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][0]["value"]

def get_sharegpt_long_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][-1]["from"] == "gpt"):
        return None
    if  len(record["conversations"]) < 6:
        return None
    return '\n'.join([c["value"] for c in record["conversations"][:-1]])

def get_llava_image_reasoning_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")

def get_llava_video_description_input(record: Dict) -> Optional[str]:
    if not (record["conversations"][0]["from"] == "human" and record["conversations"][1]["from"] == "gpt"):
        return None
    return record["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")

def get_clotho_input() -> Optional[str]:
    return "Please describe this audio."

def get_mmbench_mc_input(record: Dict) -> Optional[str]:
    q = record.get("question")
    h = record.get("hint")

    options = []
    for label in ["A", "B", "C", "D"]:
        value = record.get(label)
        if value:
            options.append(f"{label}. {value}")

    extra = (
        "Please select the correct answer from the options above. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )

    hint_line = f"Hint: {h}\n" if h else ""

    return f"{hint_line}Question: {q}\nOptions:\n" + "\n".join(options) + f"\n{extra}"

def get_llavabench_qna_input(record: Dict) -> Optional[str]:
    q = record["question"]
    extra = "Please try to answer the question with short words or phrases if possible."
    return f"{q}\n{extra}"

def get_cocoval_captioning_input() -> Optional[str]:
    return "Please describe this image in general. Directly provide the description, do not include prefix like \"This image depicts\"."

def get_videomme_mc_input(record: Dict) -> Optional[str]:
    extra = "\nThese are the frames of a video. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    q = record["question"]
    options = ast.literal_eval(record["candidates"])
    return f"{extra}\nQuestion: {q}\n" + "\n".join(options) + "\nAnswer: "


def get_mmbench_video_qna_input(record: Dict) -> Optional[str]:
    q = record["question"]
    return f"Question: {q}\nAnswer: "

def get_tempcompass_captioning_input(record: Dict) -> Optional[str]:
    return record["question"]
