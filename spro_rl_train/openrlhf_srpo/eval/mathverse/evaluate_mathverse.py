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

ds_collections = {"MathVerse_testmini": {"root": "AI4Math/MathVerse", "split": "testmini"}}

SYSTEM_PROMPT_32B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. Th answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer assistant's output should start with <answer> and end with </answer>."
SYSTEM_PROMPT_7B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."

SYSTEM_PROMPT_7B_SC = "Solve the user's question step by step. First, think about the reasoning process internally and write it inside <think> and </think> tags. Then provide the first answer in LaTeX format, wrapped with $...$, and the final result must use \boxed{}. Wrap this answer inside <answer> and </answer> tags. After that, perform a critical self-reflection on the previous reasoning and answer, writing the reflection inside <reflection> and </reflection> tags. Then, based on the reflection, generate a new reasoning process and a new answer: the new reasoning is again inside <think> and </think>, and the new answer is inside <answer> and </answer>, still using LaTeX $...$ and \boxed{}. Make sure both reasoning steps are clear and detailed. Even if the final answer does not change, the second reasoning must incorporate improvements based on the reflection. Always strictly follow the sequence: <think>...</think> <answer>...</answer> <reflection>...</reflection> <think>...</think> <answer>...</answer>. Example: <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\boxed{2}$. </answer><reflection> The reasoning is correct but too brief; I could have explained the addition more explicitly. </reflection><think> Adding $1$ and $1$ together results in $2$ because $1$ plus $1$ means taking one and adding another one, leading to $2$. </think><answer> The answer is $\boxed{2}$. </answer>. All reasoning, answer, and reflection steps must be included without omission."

SYSTEM_PROMPT_7B_SC_2 = """
Solve the user's question step by step. Begin by carefully analyzing the problem and identifying all relevant concepts, assumptions, and data. Then, think about the reasoning process internally and write it inside <think> and </think> tags. Ensure the reasoning is detailed, logically sound, and explicitly addresses each step of the problem-solving process. Next, provide the first answer in LaTeX format, wrapped with $...$, and the final result must use \boxed{}. Wrap this answer inside <answer> and </answer> tags.

After providing the answer, perform a **critical self-reflection** on the reasoning and answer, writing the reflection inside <reflection> and </reflection> tags. In the reflection, evaluate:
1. Whether the reasoning was clear, detailed, and logically sound.
2. Whether all assumptions were valid and explicitly stated.
3. Whether the final answer is accurate and consistent with the problem's requirements.
4. Whether there are alternative methods or perspectives that could improve the solution.

Based on the reflection, generate a **new and improved reasoning process** and a **new answer**. The new reasoning process should again be written inside <think> and </think> tags, incorporating all identified improvements, and the new answer should be written inside <answer> and </answer> tags, still using LaTeX $...$ and \boxed{}. Even if the final answer does not change, the second reasoning must demonstrate enhanced clarity, precision, and depth. Always strictly follow the sequence:

1. <think>...</think>
2. <answer>...</answer>
3. <reflection>...</reflection>
4. <think>...</think>
5. <answer>...</answer>

Example:

<think> Since $1+1=2$, so the answer is $2$. </think>
<answer> The answer is $\boxed{2}$. </answer>
<reflection> The reasoning is correct but too brief. It could have explained the addition more explicitly and clarified why $1+1$ equals $2$. Additionally, alternative representations, such as visualizing the addition with objects, could enhance understanding. </reflection>
<think> Adding $1$ and $1$ together results in $2$ because $1$ plus $1$ means taking one unit and adding another unit, which leads to a total of $2$. This can also be visualized by counting two objects. </think>
<answer> The answer is $\boxed{2}$. </answer>
"""


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(
            ds_collections[ds_name]["root"],
            ds_collections[ds_name]["split"],
            cache_dir=os.path.join(os.getcwd(), "data/MathVerse/"),
        )[ds_collections[ds_name]["split"]]

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            data_item["query"] = data_item["query_cot"]
            image = data_item["image"]
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT_7B},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": data_item["query"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)

        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item["image"]
            data_item["response"] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            sample_index = data_item["sample_index"]
            temp[sample_index] = data_item

        print(f"Evaluating {ds_name} ...")
        ds_name = ds_name + '_self_reflection_550_7B_prompt'
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{ds_name}_{time_prefix}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))

        # cmd = f"python mathverse/extract_calculate.py --output_file {results_file}"
        # print(cmd)
        # os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/output_ckpts/outputs_2/ckpt/global_step550_hf")
    parser.add_argument("--datasets", type=str, default="MathVerse_testmini")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model()


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/MM-EUREKA/eval/mathverse/evaluate_mathverse.py
# python mathverse/extract_calculate.py --output_file /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/MM-EUREKA/eval/results/MathVerse_testmini_self_reflection_550_7B_prompt_250509021931.json
# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org
