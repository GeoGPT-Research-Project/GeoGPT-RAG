import json
import os
import time
import datetime
import logging
from multiprocessing import Process, Queue

from langchain.vectorstores import Milvus
import openai
from pymilvus import Collection

from rag_server.embeddings import GeoEmbeddings
from rag_server.reranking import GeoReRanking
from rag_server.text_split_md import split_text
from rag_server.config import CONNECTION_ARGS, COLLECTION_NAME, CHUNK_PATH_NAME, MAX_SIZE, VEC_RECALL_NUM, \
    EXPAND_TIME_OUT, META, EXPAND_RANGE, TOP_K, SCORE_THRESHOLD, LLM_URL, LLM_KEY, RAG_PROMPT


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='rag.log', level=logging.DEBUG, format=LOG_FORMAT)
logging.FileHandler(filename='rag.log', encoding="utf-8")


if not os.path.exists(CHUNK_PATH_NAME):
    os.mkdir(CHUNK_PATH_NAME)


def meta_gen(doc):
    meta_list = []
    for t in ['title', 'section', 'subsection']:
        if len(doc.get(t, "")) > 0:
            meta_list.append(doc[t])
    return ",".join(meta_list)


def filter_docs(docs, score_threshold):
    n_docs = []
    for doc in docs:
        if "score" in doc and doc["score"] < score_threshold:
            continue
        n_docs.append(doc)
    return n_docs


def llm_generate(prompt):
    client = openai.Client(base_url=LLM_URL,
                           api_key=LLM_KEY)
    models = client.models.list()
    model = models.data[0].id
    pt = [{"role": "user", "content": prompt}]
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=pt,
                temperature=0.7,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                top_p=0.8,
                max_tokens=32768,
                timeout=3600
            )
        except ConnectionError as e:
            logging.error(f"LLM generate error: {e}")
            response = None
            time.sleep(1)
    return response.choices[0].message.content


class KBDocQA:
    def __init__(self, conn_args=CONNECTION_ARGS, col_name=COLLECTION_NAME):
        self.embeddings = GeoEmbeddings()
        self.reranking = GeoReRanking()
        self.milvus_connection_args = conn_args
        self.collection_name = col_name
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args=self.milvus_connection_args,
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            },
            drop_old=False,
            auto_id=True
        )

    def add_file(self, file_path, max_size=MAX_SIZE):
        with open(file_path, mode='r', encoding='utf-8') as f:
            text = f.read()
        chunk_data = split_text(text, file_path, max_size=max_size)
        fn = os.path.basename(file_path).split(".")[0]
        chunk_path = os.path.join(CHUNK_PATH_NAME, fn + ".jsonl")
        with open(chunk_path, mode='w', encoding='utf-8') as f:
            for d in chunk_data:
                d["chunk_path"] = chunk_path
                f.write(json.dumps(d))
                f.write("\n")
        texts = []
        metadata = []
        for d in chunk_data:
            texts.append(d["text"])
            del (d["text"])
            metadata.append(d)
        _ = self.vector_store.add_texts(
            texts,
            metadatas=metadata,
        )

    def drop_collection(self):
        if isinstance(self.vector_store.col, Collection):
            self.vector_store.col.drop()
            self.vector_store.col = None

    def vector_search(self, query, k):
        start = time.time()
        chunks = self.vector_store.similarity_search_with_score(query, k=k,
                                                                param={"metric_type": "COSINE",
                                                                       "params": {"ef": 4096}})
        results = []
        for doc, score in chunks:
            doc_info = doc.metadata
            doc_info["emb_dist"] = score
            doc_info["text"] = doc.page_content
            results.append(doc_info)
        milvus_search = time.time()
        logging.info('milvus search time: {}'.format(milvus_search - start))
        return results

    def retrieval(self, query, k=TOP_K, expand_len=EXPAND_RANGE, score_threshold=SCORE_THRESHOLD):
        plain_chunks = self.vector_search(query, k=VEC_RECALL_NUM)
        if not plain_chunks:
            return []
        top_docs = self.rerank_docs(query, plain_chunks, k, score_threshold)
        results = self.expand_docs(top_docs, expand_len)
        return results

    def rerank_docs(self, query, chunks, k, score_threshold):
        start = time.time()
        if len(chunks) == 0:
            return []

        if META:
            qp_pairs = []
            for para in chunks:
                qp_pairs.append([query, meta_gen(para) + "\n" + para['text']])
        else:
            qp_pairs = [[query, para['text']] for para in chunks]
        pred_scores = self.reranking.compute_scores(qp_pairs)
        raw_top_docs = sorted(list(zip(chunks, pred_scores)), key=lambda x: x[1], reverse=True)

        rerank_sort = time.time()
        logging.info('rerank and sort time: {}'.format(rerank_sort - start))
        return filter_docs(raw_top_docs, score_threshold)[:k]

    def expand_docs(self, docs, expand_len):
        start = time.time()
        results = []
        if expand_len > 0:
            res = []
            jobs = []
            for doc, score in docs:
                q = Queue()
                p = Process(target=self.expand_doc, args=(doc, expand_len, score, q))
                jobs.append(p)
                res.append(q)
                p.start()

            for i, q in enumerate(res):
                try:
                    n_doc = q.get(timeout=EXPAND_TIME_OUT)
                except TimeoutError as e:
                    logging.error(f'expand time out: {e}')
                    n_doc, score = docs[i]
                    n_doc.metadata["score"] = score
                    jobs[i].terminate()
                    time.sleep(0.1)
                    if not jobs[i].is_alive():
                        logging.info("[MAIN]: WORKER is a goner")
                        jobs[i].join(timeout=1.0)
                        logging.info("[MAIN]: Joined WORKER successfully!")
                q.close()
                results.append(n_doc)
        else:
            for doc, score in docs:
                doc.metadata["score"] = score
                results.append(doc)

        expand_para = time.time()
        logging.info('expand para time: {}'.format(expand_para - start))
        return results


    def _expand(self, chunks, doc, expand_range, expand_len):
        ind_list = list(chunks.keys())
        doc_len = doc["length"]
        id_set = {int(doc["index"])}
        break_flag = False
        for l in expand_range:
            if l not in chunks:
                continue
            if min(ind_list) <= l <= max(ind_list):
                if chunks[l]["length"] + doc_len > expand_len:
                    break_flag = True
                    break
                else:
                    doc_len += chunks[l]["length"]
                    id_set.add(l)
        return id_set, break_flag


    def _find_raw_place(self, chunks, doc, expand_len):
        ind_list = list(chunks.keys())
        ind = int(doc["index"])
        id_set = {int(doc["index"])}
        for k in range(1, max(max(ind_list) - int(ind), int(ind) - min(ind_list))):
            expand_range = [ind + k, ind - k]
            id_set, b_flag = self._expand(chunks, doc, expand_range, expand_len)
            if b_flag:
                break
        return sorted(list(id_set))

    def expand_doc(self, doc, expand_len, score, q):
        if not os.path.exists(doc["chunk_path"]):
            logging.info("chunk file not exist")
            doc["score"] = score
            q.put(doc)
            return
        with open(doc["chunk_path"], mode='r', encoding='utf-8') as f:
            rows = f.readlines()
        chunks = {}
        for r in rows:
            row = json.loads(r)
            if row["section"] == doc["section"]:
                chunks[int(row["index"])] = row
        if int(doc["index"]) not in list(chunks.keys()):
            doc["score"] = score
            q.put(doc)
            return
        text = ""
        id_list = self._find_raw_place(chunks, doc, expand_len)
        for i in id_list:
            if i not in chunks:
                break
            text += chunks[i]["text"]
        doc["text"] = text
        doc["score"] = score
        q.put(doc)

    def query(self, query, k=TOP_K, expand_len=EXPAND_RANGE, score_threshold=SCORE_THRESHOLD):
        docs = self.retrieval(query, k, expand_len, score_threshold)
        if len(docs) == 0:
            logging.info("retrieved 0 docs, LLM respond directly")
            resp = llm_generate(query)
        else:
            today = datetime.date.today()
            docs_text = "\n".join(
                ["[document {} begin]{}[document {} end]".format(idx, doc["text"], idx) for idx, doc in
                 enumerate(docs)])
            pt = RAG_PROMPT.replace("{search_results}", docs_text).replace("{cur_date}", str(today)).replace(
                "{question}", query)
            resp = llm_generate(pt)
        return docs, resp


