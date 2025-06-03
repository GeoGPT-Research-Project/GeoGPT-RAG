import argparse
import json
from pydantic import BaseModel
import uvicorn as uvicorn
from fastapi import FastAPI
from FlagEmbedding import FlagReranker


parser = argparse.ArgumentParser(description="deploy")
parser.add_argument('--port', default=8810, type=int)
parser.add_argument('--model_path', default="", type=str)
args = parser.parse_args()


app = FastAPI()
reranker = FlagReranker(args.model_path, use_fp16=True)


class Queries(BaseModel):
    qp_pairs: list = []


@app.post('/query')
def insert(data: Queries):
    pred_scores = reranker.compute_score(data.qp_pairs, normalize=True)
    return {"pred_scores": json.dumps(pred_scores)}


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=args.port)
