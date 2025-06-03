import requests
import json
from pydantic import BaseModel
from typing import List

from langchain.embeddings.base import Embeddings

from rag_server.config import EMBEDDING_SERVER, EMBEDDING_BATCH_SIZE


class GeoEmbeddings(BaseModel, Embeddings):

    @staticmethod
    def batch_embedding(texts: List[str]):
        t_input = {"passages": texts}

        try:
            completion = requests.post(EMBEDDING_SERVER + '/passage', json=t_input,
                                       headers={'Content-Type': 'application/json',
                                                'Connection': 'keep-alive'})
            if str(completion.status_code) != '200':
                raise ValueError(completion.content.decode('utf-8'))
            response = json.loads(completion.content.decode('utf-8'))
        except Exception as e:
            raise ConnectionError(e)

        if "q_embeddings" in response:
            if isinstance(response["q_embeddings"], list):
                embeddings = response["q_embeddings"]
            else:
                embeddings = eval(response["q_embeddings"])
        else:
            embeddings = []
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if len(texts) <= EMBEDDING_BATCH_SIZE:
            return self.batch_embedding(texts)
        embeddings = []
        for i in range(int(len(texts) / EMBEDDING_BATCH_SIZE)  + int(len(texts) % EMBEDDING_BATCH_SIZE > 0)):
            end_index = min((i + 1) * EMBEDDING_BATCH_SIZE, len(texts))
            batch_embeddings = self.batch_embedding(texts[i * EMBEDDING_BATCH_SIZE: end_index])
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        t_input = {"queries": [text]}
        try:
            completion = requests.post(EMBEDDING_SERVER + '/query', json=t_input,
                                       headers={'Content-Type': 'application/json',
                                                'Connection': 'keep-alive'})
            if str(completion.status_code) != '200':
                raise ValueError(completion.content.decode('utf-8'))
            response = json.loads(completion.content.decode('utf-8'))
        except Exception as e:
            raise ConnectionError(e)

        if "q_embeddings" in response:
            if isinstance(response["q_embeddings"], list):
                embeddings = response["q_embeddings"]
            else:
                embeddings = eval(response["q_embeddings"])
        else:
            raise ValueError("q_embeddings not in response")
        return embeddings[0]
