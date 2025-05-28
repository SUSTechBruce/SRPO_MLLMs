import os
from openai import NOT_GIVEN, OpenAI
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from .llm_base import LLMAdapter
from .types import ResponseType
from config import ENDPOINT, SUBSCRIPTION_KEY,API_VERSION
load_dotenv(override=True)

class OpenAIAdapter(LLMAdapter):
    def __init__(self, base_url: str, model: str ):
        print(f"Using OpenAIAdapter with model: {model} and base_url: {base_url} enpoint",ENDPOINT)
        self.model = model
        self.client = OpenAI(
            api_key=SUBSCRIPTION_KEY,
            base_url=base_url,
        )

    def upload_file(self, file_path: str) -> str:
        return self.client.files.create(file=Path(file_path), purpose="batch").id

    def create_batch_job(self, input_file_id: str) -> str:
        """Creates a batch job using the LLM (currently only tested with Aliyun's platform). 
        Recommended for use when operating with limited budget.

        Args:
            input_file_id (str): The ID of the file to be processed in the batch job.
            
        Returns:
            str: The ID of the created batch job.
        """
        print(f"Creating batch job with input file ID: {input_file_id}")
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        ).id

    def check_job_status(self, batch_id: str) -> str:
        return self.client.batches.retrieve(batch_id=batch_id).status

    def get_output_id(self, batch_id: str) -> str:
        return self.client.batches.retrieve(batch_id=batch_id).output_file_id

    def get_error_id(self, batch_id: str) -> str:
        return self.client.batches.retrieve(batch_id=batch_id).error_file_id

    def download_results(self, output_file_id: str, output_file_path: str):
        content = self.client.files.content(output_file_id)
        content.write_to_file(output_file_path)

    def download_errors(self, error_file_id: str, error_file_path: str):
        content = self.client.files.content(error_file_id)
        content.write_to_file(error_file_path)

    def generate(self, messages: list[dict[str, str]],type:ResponseType=None,**kwargs) -> str:
        """Generate text from the LLM (do not maintain the conversation history)
        Args:
            messages (list[dict[str, str]]): A list of message dictionaries containing role and content.
            type (ResponseType): The type of response to generate (default is 'text').
            **kwargs: Additional parameters for the chat completion request (other params you can pass to the api).
        Returns:
            str: The response from the LLM. 
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=type if type is not None else NOT_GIVEN,
            **kwargs
        )
        if type is None:
            if len(response.choices) > 1:
                return [choice.message.content for choice in response.choices]
            return response.choices[0].message.content
        else:
            if len(response.choices) > 1:
                return [choice.message.parsed for choice in response.choices]
            return response.choices[0].message.parsed
        
class AzureOpenAIAdapter(LLMAdapter):
    def __init__(self, model: str):
        from openai import AzureOpenAI
        from config import ENDPOINT, SUBSCRIPTION_KEY
        self.model = model
        self.client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=SUBSCRIPTION_KEY,
            api_version=API_VERSION,
        )

    def generate(self, messages: list[dict[str, str]], type, **kwargs) -> str:
        print(self.model,messages,type,kwargs)
        response =  self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=type if type is not None else NOT_GIVEN,
            **kwargs
        )
        if type is None:
            if len(response.choices) > 1:
                return [choice.message.content for choice in response.choices]
            return response.choices[0].message.content
        else:
            if len(response.choices) > 1:
                return [choice.message.parsed for choice in response.choices]
            return response.choices[0].message.parsed

    def upload_file(self, file_path: str) -> str:
        raise NotImplementedError("AzureOpenAIAdapter does not support file upload.")
    def create_batch_job(self, input_file_id: str) -> str:
        raise NotImplementedError("AzureOpenAIAdapter does not support batch job.")
    def check_job_status(self, batch_id: str) -> str:
        raise NotImplementedError("AzureOpenAIAdapter does not support batch job status.")
    def get_output_id(self, batch_id: str) -> str:
        raise NotImplementedError("AzureOpenAIAdapter does not support batch output id.")
    def get_error_id(self, batch_id: str) -> str:
        raise NotImplementedError("AzureOpenAIAdapter does not support batch error id.")
    def download_results(self, output_file_id: str, output_file_path: str):
        raise NotImplementedError("AzureOpenAIAdapter does not support download results.")
    def download_errors(self, error_file_id: str, error_file_path: str):
        raise NotImplementedError("AzureOpenAIAdapter does not support download errors.")
