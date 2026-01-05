from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv

def setup_openai():
    _ = load_dotenv(find_dotenv())
    
  
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    os.environ["https_proxy"] = os.getenv("https_proxy", "")
    os.environ["http_proxy"] = os.getenv("http_proxy", "")
    
    if not api_key or not base_url:
        raise ValueError("API_KEY and BASE_URL must be set in .env file")
    
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.8,
        top_p=0.95,
        openai_api_key=api_key,
        openai_api_base=base_url
    )