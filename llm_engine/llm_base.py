from abc import ABC, abstractmethod
from typing import Any

class LLMAdapter(ABC):
    @abstractmethod
    def upload_file(self, file_path: str) -> str:
        pass

    @abstractmethod
    def create_batch_job(self, input_file_id: str) -> str:
        pass

    @abstractmethod
    def check_job_status(self, batch_id: str) -> str:
        pass

    @abstractmethod
    def get_output_id(self, batch_id: str) -> str:
        pass

    @abstractmethod
    def get_error_id(self, batch_id: str) -> str:
        pass

    @abstractmethod
    def download_results(self, output_file_id: str, output_file_path: str):
        pass

    @abstractmethod
    def download_errors(self, error_file_id: str, error_file_path: str):
        pass

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        pass

