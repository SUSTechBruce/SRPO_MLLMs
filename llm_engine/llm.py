from .types import ModelType
from typing import Literal, overload, TYPE_CHECKING

PlatformType = Literal["Aliyun", "Azure", "OpenAI", "VLLM"]

if TYPE_CHECKING:
    from .local_llm import LLM as LocalLLM
    from .remote_llm import LLM as RemoteLLM

class LLM:
    """
    LLM factory class. Dynamically instantiates different LLM backends according to the `type` and `platform` arguments.

    Args:
        model (str): The model name to use.
        type (ModelType): Backend type, either "local" or "remote".
        platform (PlatformType): Remote LLM platform (Aliyun, Azure, OpenAI, VLLM), only effective when type="remote".
        **kwargs: Additional arguments passed to the specific LLM implementation.

    Example:
        - llm = LLM(type="local", model="your-local-model")
        - llm = LLM(type="remote", platform="Azure", model="your-azure-deployment")
        - llm = LLM(type="remote", platform="OpenAI", model="gpt-3.5-turbo")

    Returns:
        An LLM instance for the specified platform, with a unified interface (e.g., .generate, etc.).
    """
    @overload
    def __new__(cls, model: str=None, type: Literal["local"] = ..., **kwargs) -> "LocalLLM": ...
    @overload
    def __new__(cls, model: str=None, type: Literal["remote"]="remote", platform: PlatformType = ..., **kwargs) -> "RemoteLLM": ...
    def __new__(cls, model: str=None, type: ModelType = "local", platform: PlatformType = "", **kwargs):
        if type == "local":
            from .local_llm import LLM as LocalLLM
            return LocalLLM(model, **kwargs)
        elif type == "remote":
            from .remote_llm import LLM as RemoteLLM
            return RemoteLLM(model, platform=platform, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {type}")



