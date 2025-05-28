from typing import Literal, TypedDict, Union
from pydantic import BaseModel

Plateform = Literal["Aliyun", "Azure", "OpenAI", "VLLM"]
ModelType = Literal["local", "remote"]

class JSONSchemaType(TypedDict):
    type: Literal["json_schema"]
    json_schema: dict

class JSONObjectType(TypedDict):
    type: Literal["json_object"]

ResponseType = Union[BaseModel, JSONSchemaType, JSONObjectType]