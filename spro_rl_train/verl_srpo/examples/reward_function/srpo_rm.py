# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List
from transformers import AutoTokenizer
from mathruler.grader import grade_answer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def extract_boxed_content(text: str, last=False) -> str:
    """
    Extracts answers in \\boxed{}. 
    :param text: The input string containing the LaTeX content.
    :param last: If True, it extracts the last boxed content, otherwise the first one.
    :return: The extracted content in the first or last \boxed{}.
    """
    depth = 0
    # If last=True, find the last occurrence of \boxed{, else find the first one
    start_pos = text.rfind(r"\boxed{") if last else text.find(r"\boxed{")
    
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # Exit once we reach the corresponding closing brace
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"


def extract_token_lengths(completion):

    think_matches = re.findall(r"<think>(.*?)</think>", completion, re.DOTALL)    
    answer_matched = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL) 

    if len(answer_matched) < 2:
        return None, None

    if len(think_matches) < 2:
        return None,None

    think1 = think_matches[0].strip()
    answer1 = answer_matched[0].strip()
    
    think1 = len(tokenizer.encode(think1))
    answer1 = len(tokenizer.encode(answer1))

    L  = len(tokenizer.encode(completion))
    T_target = 2 * (think1 + answer1)
    T_max = 2.5 * (think1 + answer1)
    return L,T_target,T_max

def length_reward_func(completion):
    try:

        L,T_target,T_max = extract_token_lengths(completion)

        if L is None or T_target is None or T_max == 0:
            return 0.0

        reward = math.exp(-((abs(L - T_target)) / (T_max - T_target)) ** 2)
        # reward = math.exp(-((L - T_target) / (T_max - T_target))) ** 2
    except Exception as e:
        print(e)
    return reward

def format_reward(predict: str) -> float:
    pattern = (
        r"^(?=(?:.*<think>){2})(?=(?:.*<\/think>){2})"  # Requires exactly two <think>
        r"(?=(?:.*<answer>){2})(?=(?:.*<\/answer>){2})"  # Requires exactly two <answer>
        r"(?=(?:.*<reflection>){1})(?=(?:.*<\/reflection>){1})"  # Requires exactly one <reflection>
        r"(?!.*<think>.*<think>.*<think>)"               # Prevents three or more <think>
        r"(?!.*<\/think>.*<\/think>.*<\/think>)"          # Prevents three or more </think>
        r"(?!.*<answer>.*<answer>.*<answer>)"             # Prevents three or more <answer>
        r"(?!.*<\/answer>.*<\/answer>.*<\/answer>)"       # Prevents three or more </answer>
        r"(?!.*<reflection>.*<reflection>)"               # Prevents two or more <reflection>
        r"(?!.*<\/reflection>.*<\/reflection>)"           # Prevents two or more </reflection>
        r".*<think>(.+?)</think>\s*"                      # First <think>, non-empty
        r"<answer>(.+?)</answer>\s*"                     # First <answer>, non-empty
        r"<reflection>(.+?)</reflection>\s*"              # <reflection>, non-empty
        r"<think>(.+?)</think>\s*"                       # Second <think>, non-empty
        r"<answer>(.+?)</answer>\s*$"                    # Second <answer>, non-empty, strict end
    )
    
    matches = re.search(pattern, predict, re.DOTALL)
    return 0.5 if matches else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer_1 = extract_boxed_content(predict)
    answer_2 = extract_boxed_content(predict,last=True)
    score = 0
    if grade_answer(answer_1,ground_truth) & grade_answer(answer_2, ground_truth):
        score = 0.5 + 0.25
    elif grade_answer(answer_1,ground_truth) == True & grade_answer(answer_2,ground_truth) ==False:
        score = 0 - 0.25
    elif grade_answer(answer_1,ground_truth) == False & grade_answer(answer_2,ground_truth) ==True:
        score = 1.0
    else:
        score = 0
    return score


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.5) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        length_reward = length_reward_func(predict) * 0.1


        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + length_reward,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
