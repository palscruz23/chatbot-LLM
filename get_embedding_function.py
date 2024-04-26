from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    return embeddings
