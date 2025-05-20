import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import numpy as np
from typing import List, Dict, Tuple
import cohere
from sklearn.metrics.pairwise import cosine_similarity

from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import requests

from langchain.retrievers import EnsembleRetriever
from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class SemanticReranker:
    
    def __init__(self, cohere_api_key: str, similarity_threshold: float = 0.8):
        self.co = cohere.Client(cohere_api_key)
        self.similarity_threshold = similarity_threshold
        self.rerank_model = "rerank-english-v2.0"
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        response = self.co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return np.array(response.embeddings)
    
    def filter_by_cosine_similarity(self, 
                                   query: str, 
                                   candidates: List[str]) -> List[Tuple[str, float]]:
                                       
        if not candidates:
            return []
            
        query_embedding = self.get_embeddings([query])
        candidate_embeddings = self.get_embeddings(candidates)
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        filtered_candidates = []
        for i, (candidate, similarity) in enumerate(zip(candidates, similarities)):
            if similarity >= self.similarity_threshold:
                filtered_candidates.append((candidate, similarity))
                
        return filtered_candidates
    
    def rerank_with_cohere(self, 
                          query: str, 
                          filtered_candidates: List[Tuple[str, float]],
                          top_k: int = 5) -> List[str]:
    
        if not filtered_candidates:
            return []
        candidates = [candidate[0] for candidate in filtered_candidates]
   
        rerank_results = self.co.rerank(
            query=query,
            documents=candidates,
            model=self.rerank_model,
            top_n=top_k
        )
        

        return [result.document["text"] for result in rerank_results]
    
    def process_query(self, 
                     query: str, 
                     candidates: List[str], 
                     top_k: int = 5) -> List[str]:

        filtered_candidates = self.filter_by_cosine_similarity(query, candidates)
  
        top_contexts = self.rerank_with_cohere(query, filtered_candidates, top_k)
        
        return top_contexts


class GoogleSearchRetriver(BaseRetriever):
    k: int
    url: str
    api_key: str

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        documents = []

        headers = {
            "Content-Type": "application/json",
        }

        data = {
            'api_key': self.api_key,
            "q": query
        }

        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            for answer in result['organic']:
                document = Document(
                    page_content=answer['snippet'],
                    metadata={'link': answer['link']},
                )
                documents.append(document)
                if len(documents) == self.k:
                    break
            return documents
