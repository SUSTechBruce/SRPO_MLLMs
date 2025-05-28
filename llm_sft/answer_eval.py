import re
import asyncio
from utils.async_utils import async_run_concurrent
from llm_engine.llm import LLM
from config import DEPLOYMENT
from utils.data_utils import load_json, save_jsonl, split_batches, ensure_dir_for_file
from utils.prompt_utils import build_llm_answer_eval
from urllib.parse import urljoin
from tqdm import tqdm

def getArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEPLOYMENT, help='LLM model name or deployment name')
    parser.add_argument('--model_type', type=str, default="remote", help='LLM backend type: remote or local')
    parser.add_argument('--platform', type=str, default="VLLM", help='LLM platform: VLLM, OpenAI, Azure, etc.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--image_dir', type=str, required=False, help='Directory containing images for multimodal input')
    parser.add_argument('--image_url', type=str, default=None, help='Accessible URL prefix for images (if needed by backend)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for concurrent LLM calls')
    parser.add_argument('--concurrent_tasks', type=int, default=4, help='Number of concurrent batches to process')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Number of new results before saving a checkpoint')
    parser.add_argument('--incorrect_prefix', type=str, default='data/checkpoint/incorrect_', help='Prefix for checkpoint files of incorrect results')
    parser.add_argument('--correct_prefix', type=str, default='data/checkpoint/correct_', help='Prefix for checkpoint files of correct results')
    parser.add_argument('--incorrect_final', type=str, default='data/incorrect_final.jsonl', help='Final output file for all incorrect results')
    parser.add_argument('--correct_final', type=str, default='data/correct_final.jsonl', help='Final output file for all correct results')
    return parser.parse_args()

args = getArgs()
input_path = args.input_path
image_dir = args.image_dir
image_url = args.image_url if args.image_url else None
batch_size = args.batch_size
concurrent_tasks = args.concurrent_tasks
checkpoint_interval = args.checkpoint_interval
incorrect_prefix = args.incorrect_prefix
correct_prefix = args.correct_prefix
incorrect_final = args.incorrect_final
correct_final = args.correct_final
model = args.model
type = args.model_type
platform = args.platform

image_dir = image_url if image_url else f"file://{image_dir}"
llm = LLM(model=model, type=type, platform=platform)

records = load_json(input_path)
filtered_records = []
for rec in records:
    ans = rec.get("answer", "")
    if not isinstance(ans, str):
        continue
    ans_clean = ans.strip().upper()
    if re.fullmatch(r"[A-D]", ans_clean):
        rec["is_choice"] = True
        filtered_records.append(rec)
    elif re.fullmatch(r"-?\d+(\.\d+)?", ans_clean):
        rec["is_choice"] = False
        filtered_records.append(rec)
tqdm.write(f"Filtered records count: {len(filtered_records)}")

async def process_batch(batch_records):
    incorrect = []
    correct = []
    for rec in batch_records:
        image_url = urljoin(image_dir.rstrip('/') + '/', rec["image"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_llm_answer_eval(rec['query'])},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        try:
            response = await asyncio.to_thread(
                llm.generate,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            think = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            ans = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if not think or not ans:
                continue
            processed = {
                'query': rec['query'],
                'image': rec['image'],
                'standard_answer': rec['answer'],
                'cot': think.group(1).strip(),
            }
            raw = ans.group(1).strip()
            if rec['is_choice']:
                processed['answer'] = re.sub(r'[^A-D]', '', raw.upper())
            else:
                m = re.search(r"\d+(\.\d+)?", raw)
                processed['answer'] = m.group() if m else raw

            if processed['answer'].lower() != rec['answer'].lower():
                incorrect.append(processed)
            else:
                correct.append(processed)
        except Exception as e:
            tqdm.write(f"[LLM Error] {e}")
    return {'incorrect': incorrect, 'correct': correct}

async def process_batch_wrapper(batch):
    return await process_batch(batch)

async def main():
    all_batches = split_batches(filtered_records, batch_size)
    incorrect_all = []
    correct_all = []
    checkpoint_idx = 0

    def get_checkpoint_path(prefix, count):
        path = f'{prefix}{count}.jsonl'
        ensure_dir_for_file(path)
        return path

    def on_result(batch_result):
        nonlocal checkpoint_idx
        if batch_result:
            if batch_result.get('incorrect'):
                incorrect_all.extend(batch_result['incorrect'])
            if batch_result.get('correct'):
                correct_all.extend(batch_result['correct'])
            if len(incorrect_all) + len(correct_all) - checkpoint_idx >= checkpoint_interval:
                incorrect_path = get_checkpoint_path(incorrect_prefix, len(incorrect_all))
                correct_path = get_checkpoint_path(correct_prefix, len(correct_all))
                save_jsonl(incorrect_all, incorrect_path)
                save_jsonl(correct_all, correct_path)
                tqdm.write(f"[Checkpoint] Saved: incorrect={len(incorrect_all)}, correct={len(correct_all)}")
                checkpoint_idx = len(incorrect_all) + len(correct_all)

    try:
        await async_run_concurrent(
            all_batches,
            worker=process_batch_wrapper,
            max_concurrency=concurrent_tasks,
            on_result=on_result,
            desc="🧠 Processing batches"
        )
    except Exception as e:
        tqdm.write(f"\n[!] Exception or interruption: {e}\nSaving final results before exit...")
    finally:
        ensure_dir_for_file(incorrect_final)
        ensure_dir_for_file(correct_final)
        save_jsonl(incorrect_all, incorrect_final)
        save_jsonl(correct_all, correct_final)
        tqdm.write(f"[Final Save] Total incorrect: {len(incorrect_all)} | Total correct: {len(correct_all)}")

if __name__ == "__main__":
    asyncio.run(main())
