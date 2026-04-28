from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()   
client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY"),
    base_url="https://servicesessentials.ibm.com/apis/v3"
)

response = client.responses.create(
    model="global/anthropic.claude-sonnet-4-5-20250929-v1:0",
    input="Say hello in one sentence"
)

print(response.output[0].content[0].text)