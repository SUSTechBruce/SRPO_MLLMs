import json
import re
import argparse

DEFAULT_SOURCES = {
    "mulberry": {
        "url": "https://huggingface.co/datasets/HuanjinYao/Mulberry-SFT",
        "pattern": r"### Image Description:\n(.*?)\n\n### Rationales:"
    },
    "cot100k": {
        "url": "https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k",
        "pattern": r"<CAPTION>(.*?)</CAPTION>"
    }
}

def extract_desc(pattern, text, flags=re.DOTALL | re.IGNORECASE):
    if not text:
        return ""
    m = re.search(pattern, text, flags=flags)
    return m.group(1).strip() if m else ""

def load_and_extract(file_path, pattern, content_key="content"):
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return {
        item["image"]: extract_desc(pattern, item.get(content_key, ""))
        for item in data
    }

def main():
    parser = argparse.ArgumentParser(description="Extract image descriptions from known or custom dataset formats.")

    parser.add_argument('--source', choices=['mulberry', 'cot100k'], help='Data source name (uses default pattern).')
    parser.add_argument('--input_path', help='Your prepared data (overrides source default).')
    parser.add_argument('--pattern', help='Custom regex pattern to extract descriptions.')
    parser.add_argument('--output_path', required=True, help='Output path for image_description.jsonl.')
    parser.add_argument('--content_key', default='content', help='Key name in JSON object where text is stored.')

    args = parser.parse_args()

    if args.source:
        source_info = DEFAULT_SOURCES[args.source]
        file_path = args.input_path or f"{args.source}.jsonl"
        pattern = args.pattern or source_info["pattern"]
    else:
        if not args.input_path or not args.pattern:
            parser.error("--input and --pattern must be provided if --source is not specified.")
        file_path = args.input_path
        pattern = args.pattern

    extracted = load_and_extract(file_path, pattern, args.content_key)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for image, desc in extracted.items():
            f.write(json.dumps({"image": image, "description": desc}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

