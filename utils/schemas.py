from pydantic import BaseModel, Field

class CategoryReturnType(BaseModel):
    type: str = Field(..., description="The type of the question.")
    category: str = Field(..., description="The category of the question.")

class ReflectionSampleType(BaseModel):
    image: str = Field(..., description="The image associated with the question.")
    question: str = Field(..., description="The original question.")
    cot: str = Field(..., description="The original chain of thought.")
    answer: str = Field(..., description="The predicted answer based on CoT.")
    standard_answer: str = Field(..., description="The correct answer (ground truth).")

class ReflectionReturnType(BaseModel):
    reflection: str = Field(..., description="The reflection on the original CoT.")
