import asyncio
import os
import base64
from utils.async_utils import async_run_concurrent
from llm_engine.llm import LLM
from config import DEPLOYMENT
from utils.data_utils import load_json, split_batches, ensure_dir_for_file,save_json,load_jsonl
from utils.prompt_utils import build_llm_reflection_prompt,build_training_data_prompt,reflection_system_prompt
from utils.schemas import ReflectionReturnType
from utils.upload_utils import upload_image_and_get_url
from urllib.parse import urljoin

def getArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEPLOYMENT, help='LLM model name or deployment name')
    parser.add_argument('--model_type', type=str, default="remote", help='LLM backend type: remote or local')
    parser.add_argument('--upload_image', type=bool, required=False, help='Set to True if you want to upload images to a public image host or storage (e.g. SM.MS, Imgur, S3, R2, etc.) You need to implement the upload_image_and_get_url function in utils/upload_utils.py')
    parser.add_argument('--image_description_path', type=str, required=False, help='The path to the image description file, needed if you do not want to use the multimodal input')
    parser.add_argument('--platform', type=str, default="VLLM", help='LLM platform: VLLM, OpenAI, Azure, etc.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSONL file (incorrect samples)')
    parser.add_argument('--image_dir', type=str, required=False, help='Directory containing images for multimodal input')
    parser.add_argument('--image_url', type=str, default=None, help='Accessible URL prefix for images if you have already uploaded them to a server')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for concurrent LLM calls')
    parser.add_argument('--concurrent_tasks', type=int, default=4, help='Number of concurrent batches to process')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Number of new results before saving a checkpoint')
    parser.add_argument('--reflection_prefix', type=str, default='data/checkpoint/reflection_', help='Prefix for checkpoint files of reflection results')
    parser.add_argument('--reflection_final', type=str, default='data/reflection_final.jsonl', help='Final output file for all reflection results')
    args = parser.parse_args()
    return args

args = getArgs()
input_path = args.input_path
image_dir = args.image_dir
image_url = args.image_url if args.image_url else None
batch_size = args.batch_size
concurrent_tasks = args.concurrent_tasks
checkpoint_interval = args.checkpoint_interval
reflection_prefix = args.reflection_prefix
reflection_final = args.reflection_final
model = args.model
model_type = args.model_type
platform = args.platform
image_description_path = args.image_description_path
is_upload_image = args.upload_image

print(f"Using model: {model}, type: {model_type}, platform: {platform}",image_description_path)
images_description = load_jsonl(image_description_path) if image_description_path else None
image2desc = {item["image"]: item["description"] for item in images_description} if images_description else None
llm = LLM(model=model, type=model_type, platform=platform,syetem_prompt=reflection_system_prompt)

records = load_json(input_path)
print(f"Filtered records count: {len(records)}")

def get_image_url(rec):
    if is_upload_image:
        if "image" not in rec or not rec["image"]:
            raise ValueError(f"Record {rec} does not contain a valid 'image' field.")
        try:
            image_path = os.path.join(image_dir, rec["image"]) if image_dir else rec["image"]
            _image_url = upload_image_and_get_url(image_path)
            return _image_url
        except Exception as e:
            print(f"[Warning] Failed to upload image {rec['image']}: {e}")
            raise
    if image_url:
        return urljoin(image_url.rstrip('/') + '/', rec["image"])
    elif image_dir and platform == "VLLM":
        return urljoin(f"file://{image_dir.rstrip('/')}/", rec["image"])
    elif image_dir and platform in ("OpenAI", "Azure"):
        image_path = os.path.join(image_dir, rec["image"])
        try:
            with open(image_path, "rb") as img_f:
                img_bytes = img_f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/png;base64,{img_b64}"
        except Exception as e:
            print(f"[Warning] Failed to read or encode image {image_path}: {e}")
            return None
    else:
        return rec["image"]

async def process_batch(batch_records):
    results = []
    for rec in batch_records:
        try: 
            description = image2desc.get(rec["image"], "") if images_description else None
            prompt = build_llm_reflection_prompt(rec,description)
            messages = [
                {
                    "role": "system",
                    "content": reflection_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": get_image_url(rec)}}
                    ]
                }
            ] if image2desc is None else prompt

            response = await asyncio.to_thread(
                llm.generate,
                messages=messages,
                type=ReflectionReturnType,
            )

            reflection = response.reflection
            training_item = build_training_data_prompt(rec,reflection,rec["image"])
            results.append(training_item)
        except Exception as e:
            print(f"Error when process a batch: {e}")
            continue
            
    return results

async def process_batch_wrapper(args):
    batch = args
    return await process_batch(batch)

async def main():
    all_batches = split_batches(records, batch_size)
    all_results = []
    checkpoint_idx = 0

    def get_checkpoint_path(prefix, count):
        path = f'{prefix}{count}.json'
        ensure_dir_for_file(path)
        return path

    def on_result(batch_result):
        nonlocal checkpoint_idx
        if batch_result:
            all_results.extend(batch_result)
            if len(all_results) - checkpoint_idx >= checkpoint_interval:
                checkpoint_path = get_checkpoint_path(reflection_prefix, len(all_results))
                save_json(all_results, checkpoint_path)
                print(f"Checkpoint saved at {len(all_results)}")
                checkpoint_idx = len(all_results)
    try:
        await async_run_concurrent(
            all_batches,
            worker=process_batch_wrapper,
            max_concurrency=concurrent_tasks,
            on_result=on_result,
            desc="🧠 Processing reflection batches"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[!] Exception or interruption: {e}\nSaving final results before exit...")
    finally:
        ensure_dir_for_file(reflection_final)
        save_json(all_results, reflection_final)
        print(f"[Final Save] Total: {len(all_results)}")

if __name__ == "__main__":
    asyncio.run(main())
