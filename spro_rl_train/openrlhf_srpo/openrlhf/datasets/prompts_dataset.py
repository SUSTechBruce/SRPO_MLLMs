from torch.utils.data import Dataset
from tqdm import tqdm
import json


def preprocess_data(data, input_template=None, input_key="input", label_key="answer", apply_chat_template=None, system_prompt=None) -> str:
    # 记录是否为第一个样本
    is_first_sample = not hasattr(preprocess_data, "printed_first")
    
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            try:
                # 尝试解析JSON字符串成列表
                chat_list = json.loads(chat)
                chat = chat_list
            except:
                # 如果不是JSON格式，则处理为单个用户消息
                chat = [{"role": "user", "content": chat}]
        
        # 如果提供了系统提示且聊天中没有系统角色
        if system_prompt and not any(msg.get("role") == "system" for msg in chat):
            chat.insert(0, {"role": "system", "content": system_prompt})
            
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        
        # 如果是消息格式但没有使用chat_template，尝试处理系统提示
        if isinstance(prompt, str) and prompt.startswith("[") and system_prompt:
            try:
                chat_list = json.loads(prompt)
                if not any(msg.get("role") == "system" for msg in chat_list):
                    chat_list.insert(0, {"role": "system", "content": system_prompt})
                prompt = json.dumps(chat_list)
            except:
                pass
                
        if input_template:
            prompt = input_template.format(prompt)

    # 只打印第一个样本
    if is_first_sample:
        preprocess_data.printed_first = True
        print(f"\n===== ORIGINAL PROCESSED PROMPT (FIRST SAMPLE) =====")
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)  # 限制长度以避免输出过长
        print("===== END PROMPT =====\n")
        
        # 也打印一下原始数据，看看格式是什么样的
        print(f"\n===== ORIGINAL DATA STRUCTURE =====")
        if input_key in data:
            print(f"Input key '{input_key}': {str(data[input_key])[:500]}...")  # 打印前500个字符
        if system_prompt:
            print(f"System prompt applied: {system_prompt}")
        print("===== END DATA STRUCTURE =====\n")

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        system_prompt=None,  # 新增参数
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        
        # 获取系统提示
        self.system_prompt = system_prompt or getattr(self.strategy.args, "system_prompt", None)
        
        if self.system_prompt:
            print(f"\n===== USING SYSTEM PROMPT =====\n{self.system_prompt}\n===== END SYSTEM PROMPT =====\n")

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template, self.system_prompt)
            self.prompts.append(prompt)
            self.labels.append(label)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx]
