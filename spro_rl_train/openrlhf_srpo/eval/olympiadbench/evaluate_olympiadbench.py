import argparse
import json
import os
import random
import time

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ds_collections = {
    "OE_MM_maths_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_MM_maths_en_COMP"},
    "OE_MM_physics_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_MM_physics_en_COMP"},
    "OE_TO_maths_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_TO_maths_en_COMP"},
    "OE_TO_physics_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_TO_physics_en_COMP"},
}

SYSTEM_PROMPT_32B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. Th answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer assistant's output should start with <answer> and end with </answer>."
SYSTEM_PROMPT_7B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."


english_answer_type_dict = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}


def make_input(prompt, question_content):
    # diversified based on the vllm, which is not implemented temporarily
    input = prompt + "\n" + question_content
    return input


def build_prompt(data_item):
    is_math = "maths" in data_item["source"]
    subject_content = "Math" if is_math else "Physics"
    if data_item["is_multiple_answer"]:
        multiple_answer_text = "\\boxed{multiple answers connected with commas}"
    else:
        multiple_answer_text = "\\boxed{answer}"
    unit_text = ""
    if data_item["unit"]:
        multiple_answer_text += "(unit)"
        unit_text = ", note that the unit of the answer should not be included in \\boxed{}"
    answer_type_text = get_answer_type_text(data_item["answer_type"], multiple_answer=data_item["is_multiple_answer"])
    prompt = (
        f"The following is an open-ended problem from an International {subject_content} competition. "
        f"{answer_type_text}Please calculate the answer according to the given requirements and "
        "the information provided. Please use LaTeX format to represent the variables and formulas "
        'used in the solution process and results. Please end your solution with "So the final answer '
        f'is {multiple_answer_text}." and give the result explicitly{unit_text}.'
    )
    if is_math:
        input = make_input(prompt, data_item["question"])
    else:
        if "context" in data_item.keys() and str(data_item["context"]) != "nan":  # cannot be null
            input = make_input(prompt, data_item["context"] + "\n" + data_item["question"])
        else:
            input = make_input(prompt, data_item["question"])
    return input


def get_answer_type_text(answer_type, multiple_answer):
    # 'Tuple' has various meanings in different context, such as position or values of a series of variable,
    # so it may lead to confusion to directly use 'tuple' in the prompt.
    if ("Need_human_evaluate" in answer_type) or ("Tuple" in answer_type):
        full_answer_text = ""
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type)
            full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if "," not in answer_type:  # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type)
                full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
            else:
                answer_types = answer_type.split(",")
                answer_types = [get_single_answer_type_text(t) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
                else:
                    answer_text = ", ".join(answer_types)
                    full_answer_text = (
                        f"The problem has multiple answers, with the answers in order being {answer_text}. "
                    )
    return full_answer_text


def get_single_answer_type_text(answer_type):
    if "-" in answer_type:  # No need now
        answer_type = answer_type[: answer_type.find("-")]
    for t in ["Numerical", "Expression", "Equation", "Interval"]:
        if t in answer_type:
            return english_answer_type_dict[t]
    exit(f"Error parsing answer type {answer_type}!")


def evaluate_chat_model():
    random.seed(args.seed)

    data = []
    for ds_name in args.datasets:
        split = load_dataset(
            ds_collections[ds_name]["root"],
            ds_collections[ds_name]["split"],
            cache_dir=os.path.join(os.getcwd(), "data/OlympiadBench/"),
        )["train"]
        for data_item in split:
            data_item["source"] = ds_collections[ds_name]["split"]
            data.append(data_item)

    inputs = []
    for idx, data_item in tqdm(enumerate(data)):
        images_content = []
        if "image_1" in data_item:
            for i in range(1, 6):
                if data_item[f"image_{i}"]:
                    images_content.append({"type": "image", "image": data_item[f"image_{i}"]})

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT_32B},
                ],
            },
            {
                "role": "user",
                "content": [
                    *images_content,
                    {"type": "text", "text": build_prompt(data_item)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        if image_data:
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            )
        else:
            inputs.append(
                {
                    "prompt": prompt,
                }
            )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
    model_outputs = llm.generate(inputs, sampling_params=sampling_params)
    outputs = []
    for data_item, model_output in zip(data, model_outputs):
        if "image_1" in data_item:
            del data_item["image_1"]
            del data_item["image_2"]
            del data_item["image_3"]
            del data_item["image_4"]
            del data_item["image_5"]

        data_item["response"] = model_output.outputs[0].text
        outputs.append(data_item)

    temp = {}
    for data_item in outputs:
        id = data_item["id"]
        temp[id] = data_item

    print(f"Evaluating {ds_name} ...")
    time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
    results_file = f"{ds_name}_{time_prefix}.json"
    output_path = os.path.join(args.out_dir, results_file)
    json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print("Results saved to {}".format(output_path))

    cmd = f"python olympiadbench/extract_calculate.py --output_file {results_file}"
    print(cmd)
    # os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/output_ckpts/outputs_2/ckpt/global_step550_hf")
    parser.add_argument(
        "--datasets",
        type=str,
        default="OE_MM_maths_en_COMP,OE_MM_physics_en_COMP,OE_TO_maths_en_COMP,OE_TO_physics_en_COMP",
    )
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)

    llm = LLM(model=args.checkpoint, trust_remote_code=True, tensor_parallel_size=1, limit_mm_per_prompt={"image": 8})
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model()


# CUDA_VISIBLE_DEVICES=1 python /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/MM-EUREKA/eval/olympiadbench/evaluate_olympiadbench.py
# python mathverse/extract_calculate.py --output_file /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/MM-EUREKA/eval/results/MathVerse_testmini_self_reflection_550_7B_prompt_250509021931.json
# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org
# /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/output_ckpts/outputs_2/ckpt/global_step550_hf

