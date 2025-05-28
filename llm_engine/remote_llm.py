import asyncio
from tenacity import retry, stop_never, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor
from .adapters import OpenAIAdapter, AzureOpenAIAdapter
from .types import Plateform, ResponseType
from config import ENDPOINT, DEPLOYMENT

class LLM:
    def __init__(self, model: str, platform: Plateform = "OtherOpenAILikes", system_prompt: str = None, **kwargs):
        self.model = model
        self.adapter = self._init_adapter(platform)
        self.system_prompt = system_prompt

    def _init_adapter(self, plateform: Plateform):
        match plateform:
            case "Aliyun":
                base_url = ENDPOINT if ENDPOINT else "https://dashscope.aliyuncs.com/compatible-mode/v1"
                model = self.model if self.model else DEPLOYMENT 
                return OpenAIAdapter(base_url, model)
            case "OpenAI":
                base_url = ENDPOINT if ENDPOINT else "https://api.openai.com/v1"
                model = self.model if self.model else DEPLOYMENT 
                return OpenAIAdapter(base_url, model)
            case "VLLM":
                base_url = ENDPOINT if ENDPOINT else "http://localhost:8000/v1"
                model = self.model if self.model else DEPLOYMENT 
                return OpenAIAdapter(base_url, model)
            case "Azure":
                model = self.model if self.model else DEPLOYMENT
                return AzureOpenAIAdapter(model)
            case _:
                raise ValueError(f"Unsupported platform: {plateform}")

    async def run_batch_job(self, input_file_path: str, output_file_path: str, error_file_path: str = None):
        input_id = self.adapter.upload_file(input_file_path)
        batch_id = self.adapter.create_batch_job(input_id)

        while True:
            status = self.adapter.check_job_status(batch_id)
            if status == "completed":
                print("Batch job completed successfully.")
                break
            elif status == "failed":
                batch = self.adapter.client.batches.retrieve(batch_id=batch_id)
                print(f"Batch job failed with error: {batch.errors}")
                print("Batch job failed.")
                raise RuntimeError("Batch job failed.")
            await asyncio.sleep(5)

        self.adapter.download_results(self.adapter.get_output_id(batch_id), output_file_path)
        if error_file_path:
            error_id = self.adapter.get_error_id(batch_id)
            if error_id:
                self.adapter.download_errors(error_id, error_file_path)

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=10, max=100),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))     
    )
    def generate(self, messages: list[dict[str, str]]|str,type:ResponseType=None,**kwargs) -> str:
        """generate response (do not maintain history)
        Args:
            messages(list[dict[str, str]]|str):message, follow system, user, assistant apprroach
            type('text' | 'json_object'): the type of the response
            **kwargs: other arguments you want to pass to the adapter
        """
        try:
            if isinstance(messages, str):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
            elif isinstance(messages, list):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        }
                    ] + messages
                else:
                    messages = messages
            res = self.adapter.generate(messages, type=type, **kwargs)
        except Exception as e:
            print(f"Error generating response: {e}")
            raise         
        return res

