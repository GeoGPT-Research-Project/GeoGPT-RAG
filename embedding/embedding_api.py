from sentence_transformers import SentenceTransformer, models
import json
from pydantic import BaseModel
import uvicorn as uvicorn
from fastapi import FastAPI
import argparse
import torch

parser = argparse.ArgumentParser(description="deploy")
parser.add_argument('--port', default=8810, type=int)
parser.add_argument('--model_path', default="weights/GeoEmbedding0210", type=str)
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()


# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
app = FastAPI()
model = SentenceTransformer(args.model_path, 
                            device="cuda", 
                            trust_remote_code=True, 
                            model_kwargs=({"torch_dtype": torch.float16} if args.fp16 else None))


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


class Queries(BaseModel):
    queries: list = []
    instruction: str = task


class Passages(BaseModel):
    passages: list = []
    instruction: str = ""


@app.post('/passage')
def passage(data: Passages):
    inputs = data.passages
    instruction = data.instruction
    if instruction:
        inputs = [get_detailed_instruct(instruction, x) for x in inputs]
    vecs = model.encode(inputs, convert_to_tensor=False, normalize_embeddings=True, batch_size=32).tolist()
    return {"q_embeddings": json.dumps(vecs)}


@app.post('/query')
def query(data: Queries):
    inputs = data.queries
    instruction = data.instruction
    if instruction:
        inputs = [get_detailed_instruct(instruction, x) for x in inputs]
    vecs = model.encode(inputs, convert_to_tensor=False, normalize_embeddings=True, batch_size=32).tolist()
    return {"q_embeddings": json.dumps(vecs)}


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=args.port)
