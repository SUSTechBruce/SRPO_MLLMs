import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

import openai
import pandas as pd
import regex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_chat_response(prompt, model="gpt-4o", max_token=256, retry=5, temperature=None):
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
        if temperature is None:
            temperature = 0.5 * i
        try:
            completion = client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature, max_tokens=max_token,
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


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
"""  # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
"""  # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
"""  # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
"""  # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
"""  # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
"""  # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
"""  # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
"""  # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4]


def build_mathverse_gpt4_extract_prompt(question_data):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
"""  # noqa
    response = str(question_data["response"])
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        completion_match = regex.findall(
            r"\\boxed\{((?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*)\}", response, re.DOTALL
        )
        response = completion_match[-1][0].strip() if completion_match else response
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}7.\n{test_prompt}"

    return full_prompt


def build_mathverse_gpt4_score_prompt(question_data):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
"""  # noqa
    question_for_eval = question_data["question_for_eval"]
    extract = question_data["extraction"]
    answer = question_data["answer"]
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def post_check_score(line, prefetch=False):
    ans = str(line["answer"]).strip()
    response = str(line["extraction"]).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False


def extract_answer(problem):
    prompt = build_mathverse_gpt4_extract_prompt(problem)
    logging.info(f"sample_id: {problem['sample_index']}")
    return get_chat_response(prompt), problem["sample_index"]


def score_answer(problem):
    retry = 5
    prompt = build_mathverse_gpt4_score_prompt(problem)
    if post_check_score(problem, prefetch=True):
        res = post_check_score(problem, prefetch=True)
        logging.info(f"sample_id: {problem['sample_index']}")
        return True, problem["sample_index"]
    for i in range(retry):
        res = get_chat_response(prompt, retry=1, temperature=0.5 * i)
        if res.strip() in ["0", "1"]:
            logging.info(f"sample_id: {problem['sample_index']}")
            return int(res) == 1, problem["sample_index"]
    logging.info(f"sample_id: {problem['sample_index']}")
    return False, problem["sample_index"]


def get_acc_with_condition(res_pd, key, value):
    """
    Calculate the accuracy of predictions with a specific condition
    """
    total_pd = res_pd[res_pd[key] == value]
    correct_pd = total_pd[total_pd["score"]]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100) if len(total_pd) > 0 else "0.00"
    return len(correct_pd), len(total_pd), acc


def MathVerse_acc(result):
    total = len(result)
    correct = sum(1 for _, row in result.iterrows() if row["score"])
    accuracy = round(correct / total * 100, 2)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    df_metadata = pd.json_normalize(result["metadata"])
    result = pd.concat([result.drop("metadata", axis=1), df_metadata], axis=1)
    target_keys = ["problem_version", "subject", "subfield"]

    for key in target_keys:
        values = result[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_condition(result, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))
    pprint(scores)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_file", type=str, default="mathverse_answer.json")
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
    test_pids = list(results.keys())
    if args.number > 0:
        test_pids = test_pids[: min(args.number, len(test_pids))]
    print("Number of testing problems:", len(test_pids))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(extract_answer, results[sample_id]) for sample_id in test_pids]

        for future in as_completed(futures):
            extraction, sample_id = future.result()
            results[sample_id]["extraction"] = extraction

    print(f"Saving results to {output_file}...")
    json.dump(results, open(output_file, "w"), indent=4, ensure_ascii=False)
    print(f"Results saved.")

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(score_answer, results[sample_id]) for sample_id in test_pids]

        for future in as_completed(futures):
            score, sample_id = future.result()
            results[sample_id]["score"] = score

    print(f"Saving results to {output_file}...")
    json.dump(results, open(output_file, "w"), indent=4, ensure_ascii=False)
    print(f"Results saved.")

    results = [v for _, v in results.items()]
    scores = MathVerse_acc(pd.DataFrame(results))
    print(f"Saving scores to {result_file.replace('.json', f'_score.json')}...")
    json.dump(scores, open(result_file.replace(".json", f"_score.json"), "w"), indent=4, ensure_ascii=False)
    print(f"Scores saved.")