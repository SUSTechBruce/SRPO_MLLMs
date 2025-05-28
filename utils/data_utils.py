import re
import json
import os

def load_and_process_data(filename: str):
    with open(filename, "r") as f:
        data = load_json(f)
    for item in data:
        for message in item.get("messages", []):
            if message.get("role") == "user":
                match = re.search(r"<question>(.*?)</question>", message.get("content", ""), re.DOTALL)
                if match:
                    item["query"] = match.group(1).strip()
                else:
                    item["query"] = None
    return data

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
def save_json(records, path):
    ensure_dir_for_file(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

def split_batches(records, size):
    return [records[i:i+size] for i in range(0, len(records), size)]

def ensure_dir_for_file(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)