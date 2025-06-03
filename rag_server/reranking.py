import requests
import json
from pydantic import BaseModel
from typing import List

from rag_server.config import RERANKING_SERVER, RERANKING_BATCH_SIZE


class GeoReRanking(BaseModel):

    @staticmethod
    def batch_reranking(qp_pairs: List[List[str]]) -> List[float]:
        t_input = {"qp_pairs": qp_pairs}

        try:
            completion = requests.post(RERANKING_SERVER + '/query', json=t_input,
                                       headers={'Content-Type': 'application/json',
                                                'Connection': 'keep-alive'})
            if str(completion.status_code) != '200':
                raise ValueError(completion.content.decode('utf-8'))
            response = json.loads(completion.content.decode('utf-8'))
        except Exception as e:
            raise ValueError(e)

        if "pred_scores" in response:
            pred_scores = eval(response["pred_scores"])
        else:
            raise ValueError("reranking return error: ", response)
        if isinstance(pred_scores, float):
            pred_scores = [pred_scores]
        return pred_scores

    def compute_scores(self, qp_pairs: List[List[str]]) -> List[float]:

        if len(qp_pairs) <= RERANKING_BATCH_SIZE:
            return self.batch_reranking(qp_pairs)
        scores = []
        for i in range(int(len(qp_pairs) / RERANKING_BATCH_SIZE) + int(len(qp_pairs) % RERANKING_BATCH_SIZE > 0)):
            end_index = min((i + 1) * RERANKING_BATCH_SIZE, len(qp_pairs))
            batch_reranking = self.batch_reranking(qp_pairs[i * RERANKING_BATCH_SIZE: end_index])
            scores.extend(batch_reranking)
        return scores

