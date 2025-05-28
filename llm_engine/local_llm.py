import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM,PreTrainedModel


class LLM():
    """
    LLM class for local inference(using huggingface transformers).
    """
    def __init__(self, model: str | PreTrainedModel, system_prompt: str = "You are a helpful assistant", **kwargs):
        if isinstance(model, PreTrainedModel):
            model_name = model.name_or_path
            self.model = model
        else:
            model_name = model
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **kwargs
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.system_prompt = system_prompt
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            },
        ]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def chat(self, messages: str | list, raw: bool = False,n:int = 1):
        """Chat with the LLM (maintains the conversation history)
        Args:
            messages(str | list): The messages to chat with.
            raw(bool): Whether to return the raw output.
            n(int): The number of responses to generate.
        Returns:
            The response from the LLM.
        """
        if isinstance(messages, str):
            self.messages.append(
                {
                    "role": "user",
                    "content": messages
                }
            )
        elif isinstance(messages, list):
            for message in messages:
                self.messages.append(message)
        else:
            raise ValueError("Messages must be a string or a list of strings.")
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device=self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=n
        )
        generated = outputs[:, inputs.input_ids.shape[-1]:]

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=not raw)
        output = decoded if n > 1 else decoded[0]
        self.messages.append(
            {
                "role": "assistant",
                "content": output
            }
        )

        return output

    def generate(self, prompt: str | list, raw: bool = False,n:int = 1):
        """Generate text from the LLM (do not maintain the conversation history)
        Args:
            prompt(str | list): The prompt.
            raw(bool): Whether to return the raw output.
            n(int): The number of responses to generate.
        Returns:
            The response from the LLM.
        """
        message = [{
                    "role": "system",
                    "content": self.system_prompt
            }]
        if not isinstance(prompt, list):
            message.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        else:
            for msg in prompt:
                message.append(msg)
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device=self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=n
        )
        generated = outputs[:, inputs.input_ids.shape[-1]:]

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=not raw)

        return decoded if n > 1 else decoded[0]
    
    def get_text_with_template(self, messages: list[dict[str, str]],add_special_tokens,add_generation_prompt):
        """Get the text with the template applied"""
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens
        )
        return formatted_prompt