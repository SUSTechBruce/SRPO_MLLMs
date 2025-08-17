import unittest
class TestOpenAI(unittest.TestCase):
    def test_chat(self):
        chat()
    def test_streaming_chat(self):
        streaming_chat()
def chat():
    import openai
    base_url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
    api_version = "2024-03-01-preview"
    ak = "xxxx"  # put your key here #####
    model_name = "gpt-4o-2024-11-20"
    max_tokens = 1000  # range: [0, 4096]
    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )
    prompt = "上海天气怎么样？"
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
             {
                "role": "user",
                "content": prompt
             }
        ], 
        max_tokens=max_tokens,
        extra_headers={"X-TT-LOGID": "${your_logid}"},  # header 参数传入。请务必带上 x-tt-logid，方便定位问题。logid 生成参考：https://bytedance.larkoffice.com/wiki/wikcnF5gKiIW655Tdqux88NMloh
        #如果改模型需要thinking
        #extra_body={
        #"thinking": {
        #    "type": "enabled",
        #    "budget_tokens": 2000
        #  }
        #},
        # function_call 参数的传入方式:
        # tools=[
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "get_weather",
        #             "description": "获取某城市的天气",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "城市，如：北京"
        #                     },
        #                     "unit": {
        #                         "type": "string",
        #                         "enum": [
        #                             "celsius",
        #                             "fahrenheit"
        #                         ]
        #                     }
        #                 },
        #                 "required": [
        #                     "location"
        #                 ]
        #             }
        #         }
        #     }
        # ],
    )
    print(completion.model_dump_json())
def streaming_chat():
    import openai
    base_url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
    api_version = "2024-03-01-preview"
    ak = "y7Nqaf3gtvlVIk5wuoDG8xevRFTb5IRz"
    model_name = "gpt-4o-2024-11-20"
    max_tokens = 1000  # range: [0, 4096]
    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )
    prompt = "上海天气怎么样？"
    completion = client.chat.completions.create(
        stream=True,
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        extra_headers={"X-TT-LOGID": "${your_logid}"},  # header 参数传入。请务必带上 x-tt-logid，方便定位问题。logid 生成参考：https://bytedance.larkoffice.com/wiki/wikcnF5gKiIW655Tdqux88NMloh
        # function_call 参数的传入方式:
        # tools=[
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "get_weather",
        #             "description": "获取某城市的天气",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "城市，如：北京"
        #                     },
        #                     "unit": {
        #                         "type": "string",
        #                         "enum": [
        #                             "celsius",
        #                             "fahrenheit"
        #                         ]
        #                     }
        #                 },
        #                 "required": [
        #                     "location"
        #                 ]
        #             }
        #         }
        #     }
        # ],
    )
    for chunk in completion.response.iter_lines():
        print(chunk)


if __name__ == '__main__':
    unittest.main()