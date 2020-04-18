# -*- coding: utf-8 -*-
import sys, os, json, time
import numpy as np
from tqdm import tqdm
import random
import logging
from collections import Counter
import spacy
from spacy.tokens import Token
import copy

logger = logging.getLogger(__name__)
ANSWER_TYPE = ['PERSONNORPORG', 'PLACE', 'THING', 'TEMPORAL', 'NUMERIC']

Token.set_extension('lefts', default=[])
Token.set_extension('rights', default=[])
Token.set_extension('relative_position', default=0)
    

def tokenize(text):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    prev_is_whitespace = True
    doc_tokens = []
    
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens

def parsing_tree_dfs(node):
    N = len(node._.lefts) + len(node._.rights)
    if N == 0:
        return node.text

    text = ''
    for child in node._.lefts:
        text += parsing_tree_dfs(child)+' '
    text += node.text
    for child in node._.rights:
        text += ' '+parsing_tree_dfs(child)
    return text


def reform_tree(node):
    #print(node.text, node.head, node.text in ANSWER_TYPE)
    if node.text in ANSWER_TYPE:
        node._.lefts = []
        return True
    flag = False
    res = None
    for child in node._.lefts:
        flag |= reform_tree(child)
        if flag:
            node._.lefts.remove(child)
            node._.lefts = [child] + node._.lefts
            break
    if not flag:
        for child in node._.rights:
            flag |= reform_tree(child)
            if flag:
                node._.rights.remove(child)
                node._.lefts = [child] + node._.lefts
                break
    return flag


# 对cloze进行变化，prepend answer related words
def reformulate_quesiton(question, parser, reform_version=1):
    
    doc = parser(question)
    roots = []
    for token in doc:
        token._.lefts = [child for child in token.lefts]
        token._.rights = [child for child in token.rights]
        if token.dep_ == 'ROOT':
            roots.append(token)
        #print(token.text, token.head.text, [child for child in token.children])
    ### reformulate ###
    for root in roots:
        if reform_version == 1:
            result = reform_tree(root)
        else:
            result = False
        if result:
            roots.remove(root)
            roots = [root] + roots
    ### tree to seqence ###
    new_question = ''
    for root in roots:
        new_question += ' ' + parsing_tree_dfs(root)
    return new_question.strip()

def reformulate_demo():
    parser = spacy.load("en", disable=['ner','tagger'])
    #with open('/home/zhongli/projects/XLMRAW/data/mono/cl2/news.cl2', 'r', encoding='utf-8') as f:
    #    questions = [ line.strip().replace('PERSON/NORP/ORG' ,'PERSONNORPORG') for line in f ]
    #f = open('/home/zhongli/projects/XLMRAW/data/mono/cl2r1/news.cl2r1', 'w', encoding='utf-8')
    questions = ['What Guillermo crashed a Matt Damon interview , about his upcoming movie THING']
    qs = []
    for qu in tqdm(questions[:10], desc='reform demo'):
        tokens = qu.split(' ')
        wh = tokens[0]
        q_text = ' '.join(tokens[1:])
        print(q_text)
        q_text = reformulate_quesiton(q_text, parser, 1)
        print(q_text)
        print('----------------------')
        qu_new = wh + ' ' + q_text
        qs.append(qu_new)

def data_check(input_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    q_count = 0
    err = 0
    for entry in input_data:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                q_count += 1
                answer_text = qa['answers'][0]['text']
                answer_start= qa['answers'][0]['answer_start']
                if not context[answer_start:].startswith(answer_text):
                    err += 1
    if err == 0:
        print(input_file, 'is correct.')
    else:
        print(input_file, 'has %d problems.'%err)
    print('Number of Question:', q_count)

def data_sample_v2(input_file, sample_number, balance=False, output_file=None):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    sample_data = []
    qids = []
    q_count = 0
    
    if balance:
        whs = ['How', 'Who', 'When', 'What','Where']
        wh_qids = {}
        for wh in whs:
            wh_qids[wh] = []
        for entry in input_data:
            for paragraph in entry['paragraphs']:
                for qa in paragraph['qas']:
                    q_tokens = qa['question'].split()
                    qid = qa['id']
                    q_wh = None
                    for wh in whs:
                        if wh in q_tokens:
                            q_wh = wh
                            break
                    if q_wh is not None:
                        wh_qids[q_wh].append(qid)
        balance_number = int(sample_number / len(whs))
        for wh in whs:
            if len(wh_qids[wh]) < balance_number:
                print(wh, 'quesitons not enough.')
            random.shuffle(wh_qids[wh])
            #print(balance_number, len(balance_number))
            qids += wh_qids[wh][:balance_number]

    else:
        for entry in input_data:
            for paragraph in entry['paragraphs']:
                for qa in paragraph['qas']:
                    qids.append(qa['id'])
        random.shuffle(qids)
        qids = qids[:sample_number]

    qids = [ int(qid.split('_')[-1]) for qid in qids ]
    qids = sorted(qids)
    qids.append(-1)

    for entry in tqdm(input_data, desc="sample"):
        parags = []
        for paragraph in entry['paragraphs']:
            qas = []
            for qa in paragraph['qas']:
                qid = int(qa['id'].split('_')[-1])
                if qid == qids[q_count]:
                    qa['question'] = qa['question'].replace('PERSON/NORP/ORG', 'PERSONNORPORG')
                    qas.append(qa)
                    q_count += 1
            if len(qas) == 0:
                continue
            paragraph["qas"] = qas
            parags.append(paragraph)
        entry["paragraphs"] = parags
        sample_data.append(entry)    
    
    print('Questions Number', q_count)
    if output_file is None:
        output_file = '/'.join(input_file.split('/')[:-1]) + '/'
        if balance:
            output_file += 'balanced_'
        output_file += 'sample_%dw-'%(int(sample_number/10000))+input_file.split('/')[-1]
    print('Saving to',output_file)
    json.dump({"version": "v2.0", 'data': sample_data}, open(output_file ,'w',encoding='utf-8'))

def filter_data_given_qids(input_data_, qids, is_sorted=False):
    input_data = copy.deepcopy(input_data_)
    if not is_sorted:
        qids = sorted(qids, key=lambda x: int(x.strip().split('_')[-1]))
    q_count = 0
    new_data = []
    for entry in tqdm(input_data, desc='filter'):
        paras = []
        for paragraph in entry['paragraphs']:
            qas = []
            for qa in paragraph['qas']:
                if q_count < len(qids) and qa['id'] == qids[q_count]:
                    qas.append(qa)
                    q_count += 1
            if len(qas) == 0:
                continue
            paragraph['qas'] = qas
            paras.append(paragraph)
        if len(paras) == 0:
            continue
        entry['paragraphs'] = paras
        new_data.append(entry)
    return new_data   

def data_split(input_file, data_size):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    sample_data = []
    qids = []
    q_count = 0
    for entry in input_data:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                qids.append(qa['id'])
    random.shuffle(qids)

    num = 0
    while num*data_size < len(qids):
        nqids = qids[num*data_size: min(len(qids), (num+1)*data_size)]
        new_data = filter_data_given_qids(input_data, nqids)
        output_file = '/'.join(input_file.split('/')[:-1]) + '/'
        output_file += ('%d_'%num)+input_file.split('/')[-1]
        json.dump({"version": "v2.0", 'data': new_data}, open(output_file ,'w',encoding='utf-8'))
        print(output_file, len(nqids))
        data_check(output_file)
        num += 1


def data_concat(files, output_file):
    all_data = []
    for input_file in files:
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
            all_data += input_data        
    json.dump({"version": "v2.0", 'data': all_data}, open(output_file ,'w',encoding='utf-8'))

def split_all_data(data_dir, input_file, output_files, output_sizes):
    input_file = os.path.join(data_dir, input_file)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    qids = []
    for entry in input_data:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                qids.append(qa['id'])

    q_pos = 0
    assert len(output_files) == len(output_sizes)

    for output_file, data_size in zip(output_files, output_sizes):
        output_file = os.path.join(data_dir, output_file)
        nqids = qids[q_pos: q_pos+data_size]
        q_pos += data_size
        new_data = filter_data_given_qids(input_data, nqids)
        json.dump({"version": "v2.0", 'data': new_data}, open(output_file ,'w',encoding='utf-8'))
        data_check(output_file)

def recover_wikiref():
    input_file = "../uqa_all_data.json"
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger'])

    wikiref_data = {}

    for entry in tqdm(input_data, desc="Recover Wikiref"):
        title = entry['title']
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            doc = nlp(context[len(title):].strip())
            sents = [title] + [sent.text.strip() for sent in doc.sents]
            for qa in paragraph['qas']:
                qid = qa['id']
                summary = qa['summary']
                uid = qid.split('_')[0]
                wikiref_data[uid] = {
                    "uid": uid,
                    "document": sents,
                    "summary": summary
                }
    wikiref = [wikiref_data[key] for key in wikiref_data]
    print(len(wikiref))
    json.dump(wikiref, open("../wikiref.json", "w", encoding='utf-8'), indent=4)

if __name__ == '__main__':
    split_all_data('../', 'uqa_all_data.json', 
        ['uqa_train_main.json'] + ['uqa_train_refine_%d.json'%i for i in range(6)], 
        [300000] + [100000 for _ in range(6)])

