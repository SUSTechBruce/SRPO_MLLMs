import argparse
import copy as cp
import json
import logging
import os
import re
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd
import regex
from latex2sympy2 import latex2sympy
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_chat_response(prompt, model="gpt-4o", max_token=256, retry=5):
    messages = [
        {"role": "user", "content": prompt},
    ]

    base_url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
    api_version = "2024-03-01-preview"
    ak = "xxxx"  # put your key here #####
    model_name = "gpt-4o-2024-11-20"


    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )
    for i in range(retry):
        try:
            completion = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.5 * i, max_tokens=max_token,
                extra_headers={"X-TT-LOGID": "${your_logid}"},
            )
            prediction = completion.choices[0].message.content.strip()
            if prediction != "" and prediction is not None:
                return prediction
            else:
                continue
        except Exception as e:
            logging.error(e)
    return ""


def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print("Warning: input is not string")
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False


def can_infer_option(answer, choices):
    verbose = os.environ.get("VERBOSE", 0)
    # Choices is a dictionary

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3 and verbose:
                print(f"A might be a quantifier in the string: {answer}.")
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Please solve the problem step by step and put your answer in one \"\\boxed{}\". If it is a multiple choice question, only one letter is allowed in the \"\\boxed{}\".\n
How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs \\boxed{1.45$} to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Please solve the problem step by step and put your answer in one \"\\boxed{}\". If it is a multiple choice question, only one letter is allowed in the \"\\boxed{}\".\n
What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is \\boxed{B}.\n
Extracted answer: B
"""
    return [example_1, example_2, example_3, example_4, example_5]


def build_mathv_gpt4_prompt(question_data):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    options = ""
    if len(question_data["options"]) > 0:
        assert len(question_data["options"]) == 5, question_data
        if "".join(question_data["options"]) != "ABCDE":
            options = f"(A) {question_data['options'][0]}\n(B) {question_data['options'][1]}\n(C) {question_data['options'][2]}\n(D) {question_data['options'][3]}\n(E) {question_data['options'][4]}\n"
    question = f"{question_data['question']}\n{options}"
    response = str(question_data["response"])
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        completion_match = regex.findall(
            r"\\boxed\{((?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*)\}", response, re.DOTALL
        )
        response = completion_match[-1][0].strip() if completion_match else response
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + "\n"
    prompt += question + "\n"
    prompt += "Model respone: " + response + "\n"
    prompt += "Extracted answer:"
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def post_check(question_data, prefetch=False):
    res = None
    ans = question_data["answer"]
    response = question_data["response"] if prefetch else question_data["extraction"]
    try:
        if len(question_data["options"]) > 0:
            ans = question_data["answer"]
            choices = list_to_dict(question_data["options"])
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass
    if is_equal(res, ans):
        return res if prefetch else True
    else:
        return False


def extract_answer(problem):
    prompt = build_mathv_gpt4_prompt(problem)
    if post_check(problem, prefetch=True):
        res = post_check(problem, prefetch=True)
        logging.info(f"id: {problem['id']}")
        return res, problem["id"]
    logging.info(f"id: {problem['id']}")
    return get_chat_response(prompt), problem["id"]


def MATH_V_acc(result):
    data = result
    tot = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data[i]
        cate = item["subject"]
        tot["Overall"] += 1
        tot[cate] += 1
        if post_check(item, prefetch=False):
            hit["Overall"] += 1
            hit[cate] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res["Subject"].append(k)
        res["tot"].append(tot[k])
        res["hit"].append(hit[k])
        res["acc"].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res).sort_values("Subject", ignore_index=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--response_label", type=str, default="response", help="response label for the input file")
    parser.add_argument("--number", type=int, default=-1, help="number of problems to run")
    parser.add_argument("--output_label", type=str, default="extract", help="label for the output file")
    args = parser.parse_args()

    # args
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != "":
        output_file = result_file.replace(".json", f"_{args.output_label}.json")
    else:
        output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = json.load(open(result_file))

    # full pids
    test_ids = list(results.keys())
    if args.number > 0:
        test_ids = test_ids[: min(args.number, len(test_ids))]
    print("Number of testing problems:", len(test_ids))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(extract_answer, results[sample_id]) for sample_id in test_ids]

        for future in as_completed(futures):
            extraction, id = future.result()
            results[id]["extraction"] = extraction

    print(f"Saving results to {output_file}...")
    json.dump(results, open(output_file, "w"), indent=4, ensure_ascii=False)
    print(f"Results saved.")

    results = [v for _, v in results.items()]
    scores = MATH_V_acc(results)
    print("\n" + tabulate(scores))
    print(f"Saving scores to {result_file.replace('.json', f'_score.json')}...")
    json.dump(
        json.loads(scores.to_json(orient="records")),
        open(result_file.replace(".json", f"_score.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )
    print(f"Scores saved.")