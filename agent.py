from dotenv import load_dotenv
import os
import openai
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

from collections import deque
from typing import Dict, List, Optional, Any

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

# Vector Store 셋업
from langchain.vectorstores import FAISS # 고차원 벡터 검색에 최적화된 라이브러리
from langchain.docstore import InMemoryDocstore

# Define your embedding model
embeddings_model = OpenAIEmbeddings() # OpenAi 임베딩 모델 선택
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536 # OpenAI의 임베딩 모델이 1536차원 벡터를 생성하도록 되어 있어 맞는 사이즈 설정
index = faiss.IndexFlatL2(embedding_size) 
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})