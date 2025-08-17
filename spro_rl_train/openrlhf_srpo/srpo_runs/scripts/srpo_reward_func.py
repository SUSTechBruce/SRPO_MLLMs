import os
import re
from datetime import datetime
from transformers import AutoTokenizer
import math

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d"]
problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<|end▁of▁sentence|>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def get_query_from_query(q: str):
    try:
        matches = re.findall(problem_pattern, q, re.DOTALL)
        return matches[0]
    except:
        return q


# def extract_answer_with_tags(text):
#     match = re.search(r"(<answer>.*?</answer>)", text)
#     if match:
#         return match.group(1)
#     return None

def extract_answer_with_tags(text):
    matches = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
    
    if not matches:
        return None, None
    elif len(matches) == 1:
        return matches[0], matches[0], matches
    else:
        return matches[0], matches[-1], matches # return first response and second response

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

def accuracy_reward_func(completion, answer):
    reward = 0.0
    response_1, response_2, matches = extract_answer_with_tags(completion)

    if response_1 is not None and response_2 is not None: # have first answer and and second answer
        response_1 = response_1
        response_2 = response_2
    else:
        try:
            response_1 = completion.split("<answer>")[-1] # set to the same answer if extract_answer wrong
            response_2 = completion.split("<answer>")[-1]

        except:
            response_1 = completion.split("\n")[-1] # set to the same answer if extract_answer wrong
            response_2 = completion.split("\n")[-1]

    content_1, sol = response_1, answer
    content_2, sol = response_2, answer
    answer_parsed_1 = content_1
    answer_parsed_2 = content_2

    sol = f"${str(sol)}$"
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        answer_parsed_1 = parse(
            content_1,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )

        answer_parsed_2 = parse(
            content_2,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )

        if len(matches) == 1: # only have first answer

            try:

                if verify(answer_parsed_1, gold_parsed):
                    reward = 0.5 
            except Exception:
                pass

        else:  # get first and the second answer

            try:

                if verify(answer_parsed_1, gold_parsed) == False and verify(answer_parsed_2, gold_parsed) == True:
                    reward = 0.5 + 0.5 # correct the wrong answer after relfection

                elif verify(answer_parsed_1, gold_parsed) and verify(answer_parsed_2, gold_parsed): # both truth
                    reward = 0.5 + 0.25 #  keep the corrected answer 

                elif verify(answer_parsed_1, gold_parsed) == True and verify(answer_parsed_2, gold_parsed) == False:

                    reward = 0.5 - 0.25 #  mislead the corrected answer 

                else: # change nothing

                    reward = 0.0

                # reward = float(verify(answer_parsed, gold_parsed))
            except Exception:
                pass

        if reward == 0.0: # only check the final answer after reflection
            try:


                # content_match = re.search(r"<answer>(.*?)</answer>", completion)

                content_match_1, content_match_2, matches = extract_answer_with_tags(completion)
                content_match = content_match_2
 
                # student_answer = content_match.group(1).strip() if content_match else content_2.strip()

                student_answer = content_match.strip() if content_match else content_2.strip() # using findall not search


                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed_1 # not use

def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){2})(?=(?:.*<\/think>){2})"  # Require exactly two <think>
        r"(?=(?:.*<answer>){2})(?=(?:.*<\/answer>){2})"  # Require exactly two <answer>
        r"(?=(?:.*<reflection>){1})(?=(?:.*<\/reflection>){1})"  # Require exactly one <reflection>
        r"(?!.*<think>.*<think>.*<think>)"               # Prevent three or more <think>
        r"(?!.*<\/think>.*<\/think>.*<\/think>)"        # Prevent three or more </think>
        r"(?!.*<answer>.*<answer>.*<answer>)"           # Prevent three or more <answer>
        r"(?!.*<\/answer>.*<\/answer>.*<\/answer>)"     # Prevent three or more </answer>
        r"(?!.*<reflection>.*<reflection>)"             # Prevent two or more <reflection>
        r"(?!.*<\/reflection>.*<\/reflection>)"         # Prevent two or more </reflection>
        r".*<think>(.+?)</think>\s*"                    # The first think, non-empty
        r"<answer>(.+?)</answer>\s*"                   # The first answer, non-empty
        r"<reflection>(.+?)</reflection>\s*"            # The reflection, non-empty
        r"<think>(.+?)</think>\s*"                     # The second think, non-empty
        r"<answer>(.+?)</answer>\s*$"                  # The second answer, non-empty, strictly at the end
    )

    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0

def reward_func(queries, prompts, labels):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    length_rewards = []
    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, answer in zip(queries, prompts, labels):
            try:
                response = get_response_from_query(query)
                if response == "":
                    f.write("Error: " + query + "\n")
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    length_rewards.append(0.0)
                    format_rewards.append(0.0)

                else:
                    query1 = get_query_from_query(query)

                    
                    accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func(response)
                    length_reward = length_reward_func(response) * 0.1

                    rewards.append(accuracy_reward + format_reward + length_reward)
                    accuracy_rewards.append(accuracy_reward)
                    length_rewards.append(length_reward)
                    format_rewards.append(format_reward)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1 + "\n")
                    f.write("Response: " + response + "\n")
                    f.write("Answer: " + answer + "\n")
                    f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                length_rewards.append(0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
        "length_rewards": torch.tensor(length_rewards, dtype=torch.float32)
    }