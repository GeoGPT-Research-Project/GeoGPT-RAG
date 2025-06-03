import re
from collections import defaultdict

import math
import torch
from nltk import sent_tokenize
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

from rag_server.config import BERT_PATH


SECTIONS = ['abstract', 'introduction', 'conclusion', 'acknowledgement', 'reference']
FILTERED_SECTIONS = ['acknowledgement', 'reference', 'acknowledgment']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_model(model_name=BERT_PATH):
    tokz = BertTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    model = model.to(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    return tokz, model


tokz, model = set_model()


def model_predict(model,
                  input_ids,
                  attention_mask,
                  token_type_ids, batch_size=64):
    try:
        predict_result = model_result(model,
                                      input_ids,
                                      attention_mask,
                                      token_type_ids, batch_size)
    except RuntimeError:
        torch.cuda.empty_cache()
        try:
            predict_result = model_result(model,
                                          input_ids,
                                          attention_mask,
                                          token_type_ids, batch_size // 2)
        except RuntimeError:
            predict_result = model_result(model,
                                          input_ids,
                                          attention_mask,
                                          token_type_ids, batch_size // 4)

    return predict_result


def model_result(model,
                 input_ids,
                 attention_mask,
                 token_type_ids, batch_size):
    model.eval()
    with torch.no_grad():
        all_probs = []
        batch_num = math.ceil(len(input_ids) / batch_size)
        for i in range(batch_num):
            start = batch_size * i
            end = min(start + batch_size, len(input_ids))

            t_input_ids = torch.tensor(input_ids[start:end]).to(device)
            t_attention_mask = torch.tensor(attention_mask[start:end]).to(device)
            t_token_type_ids = torch.tensor(token_type_ids[start:end]).to(device)

            outputs = model(input_ids=t_input_ids,
                            attention_mask=t_attention_mask,
                            token_type_ids=t_token_type_ids)
            logits = outputs.logits
            probs = softmax(logits, dim=1).tolist()
            all_probs.extend(probs)

    predict_result = [[prob.index(max(prob)), prob[0]] for prob in all_probs]
    return predict_result


def reset_chunk(new_chunks, sentences, max_token_size):
    if len(sentences) < 1:
        return
    if len(sentences) == 1:
        new_chunks.append(sentences)
        return
    all_sen = 0
    scores = []
    for i, sen, t_size, score in sentences:
        all_sen += t_size
        scores.append(score)
    if all_sen <= max_token_size:
        new_chunks.append(sentences)
        return

    new_score = scores[1:]
    min_score = min(new_score)
    ind = new_score.index(min_score) + 1

    r_sens = sentences[:ind]
    l_sens = sentences[ind:]

    reset_chunk(new_chunks, r_sens, max_token_size)
    reset_chunk(new_chunks, l_sens, max_token_size)


def sentence_counter(text, counter):
    for c in text:
        for label in counter:
            if c == label:
                counter[label] += 1
    return counter


def concat_sentences(sentences):
    n_sentences = []
    temp = []
    counter = {"(": 0, "[": 0, ")": 0, "]": 0}
    for sent in sentences:
        counter = sentence_counter(sent, counter)
        temp.append(sent)
        if (counter["("] > counter[")"] or counter["["] > counter["]"]) and len(temp) < 10:
            continue
        else:
            n_sentences.append(temp)
            temp = []
            counter = {"(": 0, "[": 0, ")": 0, "]": 0}
    return [" ".join(s) for s in n_sentences]


def merge_chunks(chunks, max_token_size):
    prev_sum_size = 0
    n_chunks = []
    cur_chunk = []
    for rc in chunks:
        total_size = sum([int(t) for _, _, t, _ in rc])
        if prev_sum_size + total_size > max_token_size:
            prev_sum_size = total_size
            if len(cur_chunk) > 0:
                n_chunks.append(cur_chunk)
            cur_chunk = rc
        else:
            cur_chunk.extend(rc)
            prev_sum_size = prev_sum_size + total_size
    n_chunks.append(cur_chunk)
    if len(n_chunks) == 0 or len(n_chunks[0]) == 0:
        print(chunks)
        print(n_chunks)
        print(cur_chunk)
    return n_chunks


def split_long_para(text, tokz, model, max_token_size):
    sentences = sent_tokenize(text)
    sentences = concat_sentences(sentences)
    sentences = [s.strip() for s in sentences if s]
    if len(sentences) <= 1:
        return [(s, len(tokz(s).input_ids)) for s in sentences]

    results = [[1, 1]] * len(sentences)
    tok_sizes = [len(tokz(s).input_ids) for s in sentences]
    former_sen = sentences[:len(sentences) - 1]
    latter_sen = sentences[1:]

    tokens = tokz(former_sen, latter_sen, padding='max_length', max_length=512, return_tensors=None)
    input_ids = []
    attention_mask = []
    token_type_ids = []
    sent_ids = []
    for i, input_id in enumerate(tokens.input_ids):
        if len(input_id) > 512:
            continue
        sent_ids.append(i + 1)
        input_ids.append(input_id)
        attention_mask.append(tokens.attention_mask[i])
        token_type_ids.append(tokens.token_type_ids[i])
    model_result = model_predict(model, input_ids, attention_mask, token_type_ids)

    assert len(model_result) == len(sent_ids)
    for i, mr in zip(sent_ids, model_result):
        results[i] = mr

    predict_chunks = []
    for i, (sen, t_size, (relation, score)) in enumerate(zip(sentences, tok_sizes, results)):
        if len(predict_chunks) == 0:
            predict_chunks.append([[i, sen, t_size, score]])
        else:
            predict_chunks[-1].append([i, sen, t_size, score])

    reset_chunks = []
    for sentence_info in predict_chunks:
        temp_chunks = []
        reset_chunk(temp_chunks, sentence_info, max_token_size)
        reset_chunks.extend(temp_chunks)

    reset_chunks = merge_chunks(reset_chunks, max_token_size)
    reset_chunks = [sorted(tc, key=lambda x: x[0]) for tc in reset_chunks]
    reset_chunks = sorted(reset_chunks, key=lambda x: x[0][0])
    out_chunks = [(" ".join([s for _, s, _, _ in rc]), sum([int(t) for _, _, t, _ in rc])) for rc in reset_chunks]
    return out_chunks


def cal_size(section):
    num = 0
    for c in section.strip():
        if c == '#':
            num += 1
        else:
            break
    return num


def check_section(text):
    for i in SECTIONS:
        if i in text:
            return True
    return False


def reset_section(sections_dict):
    # 如果都没有识别主章节，将副章节名称移到主章节名称
    empty = True
    for i, sections in sections_dict.items():
        if sections[0] != '':
            empty = False
    if empty:
        n_sections = {}
        for i, sections in sections_dict.items():
            n_sections[i] = (sections[1], '')
        return n_sections
    else:
        return sections_dict


def sentence_close(text):
    # 判断句子是否已经结束
    left_num = 0
    right_num = 0
    for c in text:
        if c == '(':
            left_num += 1
        if c == ')':
            right_num += 1
    if right_num < left_num:
        return False
    else:
        return True


def filter_section(text):
    # 去除 Reference 等信息
    for i in FILTERED_SECTIONS:
        if i in text.strip().lower():
            return True
    return False


def _concat(prev, new):
    if prev[-1] == '-':
        prev = prev + new
    else:
        prev = prev + ' ' + new
    return prev


def concat_ele(ele, n_ls):
    # 如果前一句判断没有结束，则直接拼接
    if n_ls != [] and (n_ls[-1][-1] != '.' or not sentence_close(n_ls[-1])):
        n_ls[-1] = _concat(n_ls[-1], ele)
        return n_ls
    # 如果前后两段都特别短，则直接拼接
    if n_ls != [] and (len(n_ls[-1]) < 100 or len(ele) < 100):
        n_ls[-1] = n_ls[-1] + "\n" + ele
        return n_ls
    n_ls.append(ele)
    return n_ls


def merge_ele(ls):
    n_ls = []
    tab_start = False
    for ele in ls:
        if len(ele) == 0:
            continue
        # 处理表格，将表格信息拼接一起
        if re.search(r'\|(.+)\|', ele, re.M | re.I):
            if not tab_start:
                n_ls.append(ele)
                tab_start = True
                continue
            else:
                n_ls[-1] = n_ls[-1] + '\n' + ele
                continue
        elif tab_start:
            tab_start = False
            n_ls.append(ele)
            continue

        n_ls = concat_ele(ele, n_ls)
    return n_ls


def split_para(text):
    # 先按 \n 分段
    ls_0 = [i.strip() for i in text.split("\n") if i]
    err = []
    ls = []
    for i in ls_0:
        if i == "":
            continue
        if not i.startswith("#") and i in ls:
            err.append(i)
        else:
            ls.append(i)
    return ls


def extract_title(info, ele):
    # 提取标题
    processed = False
    if cal_size(ele) == 1:
        info["title"] = ele.lstrip('#').strip()
        info["s_index"] = 0
        info["section"] = ""
        info["subsection"] = ""
        info["sections"] = {0: ('', '')}
        info["raw_dict"] = defaultdict(list)
        processed = True
    return info, processed


def extract_section(info, match, ele):
    # 提取主章节名称
    processed = False
    if match and (check_section(ele.lstrip('#').strip().lower()) or cal_size(ele)) == 2:
        info["section"] = ele.lstrip('#').strip()
        info["subsection"] = ''
        info["s_index"] += 1
        info["sections"][info["s_index"]] = (info["section"], info["subsection"])
        processed = True
    return info, processed


def extract_subsection(info, b_match, b_text, ele):
    # 提取副章节名称
    processed = False
    if cal_size(ele) == 3 or b_text:
        if b_text:
            m, n = b_match
            info["subsection"] = ele[m:n]
            ele = ele[n:].strip()
        else:
            info["subsection"] = ele.lstrip('#').strip()
            ele = ""
        info["s_index"] += 1
        info["sections"][info["s_index"]] = (info["section"], info["subsection"])
        if ele != "":
            info["raw_dict"][info["s_index"]].append(ele)
        processed = True
    return info, processed


def extract_outlines(ls):
    # 将文本结构化拆分
    info = {
        "raw_dict": defaultdict(list),
        "sections": {0: ('', '')},
        "title": "",
        "s_index": 0,
        "section": "",
        "subsection": ""
    }

    for i, ele in enumerate(ls):
        b_match = re.search(r'\*{2}.{2,}\*{2}', ele, re.M | re.I)
        match = re.search(r'#+ ', ele, re.M | re.I)
        if b_match and b_match.span()[0] == 0 and b_match.span()[1] < 50:
            b_text = True
        else:
            b_text = False
        if match or b_text:
            # 提取标题
            info, p_flag = extract_title(info, ele)
            if p_flag:
                continue
            # 提取主章节名称
            info, p_flag = extract_section(info, match, ele)
            if p_flag:
                continue
            # 提取副章节名称
            info, p_flag = extract_subsection(info, b_match, b_text, ele)
            if p_flag:
                continue
        info["raw_dict"][info["s_index"]].append(ele)
    return info


def refine_structure(info):
    # 移除一些空章节，将章节内段落进行有效合并
    sections = reset_section(info["sections"])
    s_dict = {}
    all_s = []
    for i in sections:
        if i in info["raw_dict"]:
            if filter_section(sections[i][0]):
                break
            s_list = merge_ele(info["raw_dict"][i])
            s_dict[i] = s_list
            all_s.extend(s_list)
    return sections, s_dict, all_s



def split_text(text, fn, max_size=512):
    ls = split_para(text)
    info = extract_outlines(ls)
    sections, s_dict, all_s = refine_structure(info)    # 将文本结构化分段

    data = []
    r_index = 0
    for i, s_list in s_dict.items():
        all_para = "\n".join(s_list)
        total_size = len(tokz(all_para).input_ids)
        # 较长段落进行 Bert 分段
        if total_size > max_size:
            new_eles = split_long_para(all_para, tokz, model, max_size)
            n_paras = new_eles
        else:
            n_paras = [(all_para, total_size)]
        for para, t_size in n_paras:
            data.append({'title': info["title"], 'section': sections[i][0], "source": fn,
                         'subsection': sections[i][1], 'index': r_index, 'text': para, "length": t_size})
            r_index += 1

    return data

