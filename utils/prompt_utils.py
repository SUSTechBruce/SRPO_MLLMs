from .schemas import ReflectionSampleType

reflection_system_prompt = "You are a helpful math reasoning assistant. Think carefully. Output only JSON."
category_system_prompt = "You are a helpful question classifier model. Output only JSON."

def build_catergory_prompt(query: str) -> str:
    prompt =  f'''
Please follow this structure strictly:
"category": Identify the main category of the questions. Use one of the following categories:
    - "Mathematical Reasoning"
    - "Physical Reasoning"
    - "Chemical Reasoning"
    - "Natural Scene Reasoning"
    - "Logical Puzzle Reasoning"
    - "Other" (if the question does not fit into the above categories, specify the category you believe it belongs to)
"type": Actual type of the question like  - "Statistics" - "Geometry" - "Algebra"

Only return a valid JSON object with "category" and "type" fields.

--- Input ---
Question:
{query}

—--- Output ---
{{
    "category": "Mathematical Reasoning",
    "type": "Algebra"
}}
'''
    return prompt

def build_llm_reflection_prompt(sample:ReflectionSampleType,description:str | None = None) -> str:
    prompt =f"""You are an expert visual reasoning assistant. Your task is to reflect on the quality of a chain-of-thought (CoT) reasoning given for a visual question. The goal is to **improve** the CoT by identifying weaknesses and offering suggestions for refinement.

Please follow this structure strictly:
1. "reflection": Provide a detailed critique of the original CoT, pointing out:
    - Logical flaws or inconsistencies
    - Missing assumptions or information
    - Any correct reasoning that could be made clearer or more robust
    - Suggestions for improving the reasoning process

Only return a valid JSON object with a **reflection** field.

--- Input ---
Question:
{sample['query']}

Original Chain of Thought:
{sample['cot']}

Predicted Answer (Based on CoT):
{sample['answer']}

Correct Answer (ground truth):
{sample['standard_answer']}

"""
    if description:
        prompt += f'\nImage Description: {description}\n'
    return prompt
def build_llm_answer_eval(query: str) -> str:
    return f"""You are a precise AI assistant and must strictly follow the following rules:\n" \
                    "1. First reason step-by-step, and wrap the thought process in <think> tags\n" \
                    "2. The final answer must be wrapped in <answer> tags\n" \
                    "3. Formatting requirements:\n" \
                    "   - Choice answers must be uppercase letters (A/B/C/D)\n" \
                    "   - Fill-in-the-blank answers should be digits\n" \
                    "DO NOT EXPLAIN ANYTHING IN <answer></answer>\n" \
                    "You must provide both <think> and <answer>.\n" \
                    "Strictly follow the format, do not add anything extra!" \
                    Question: {query}
                    """


def build_training_data_prompt(rec1, rec2, image_url: str) -> dict:
    instruction = (
    "You are a reasoning expert. Given an image and a question, please generate two rounds of step-by-step reasoning: First, provide your initial chain of thought and answer. Then reflect on it, and finally, based on your reflection, give your final reasoning and answer."
)
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {
            "role": "user",
            "content": (
                f"Question:<question>{rec1['query']}</question>\n"
                f"Image: <image>\n"
            )
        },
        {
            "role": "assistant",
            "content": f"<think>{rec1['cot']}</think>"
            f"\n<answer>{rec1['answer']}</answer>"
            f"\n<reflection>{rec2}</reflection>\n"
            f"<answer>{rec1['standard_answer']}</answer>"
        }
    ]

    images = [image_url + rec1["image"]]

    return{
        "messages": messages,
        "images": images
    }