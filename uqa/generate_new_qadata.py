# -*- coding: utf-8 -*-
import sys, os, json, time
import numpy as np
from tqdm import tqdm
import random
import logging
from collections import Counter
import spacy
import copy
import benepar
import nltk
from data_utils import reformulate_quesiton, data_check, filter_data_given_qids
from cloze2natural import identity_translate
from wikiref_process import get_clause_v2
import argparse

nltk.download('punkt')
benepar.download('benepar_en2')

entity_category = {
    'PERSONNORPORG' : "PERSON, NORP, ORG".replace(' ','').split(','),
    'PLACE' : "GPE, LOC, FAC".replace(' ','').split(','),
    'THING' : 'PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE'.replace(' ','').split(','),
    'TEMPORAL': 'TIME, DATE'.replace(' ','').split(','),
    'NUMERIC' : 'PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL'.replace(' ','').split(',')
}
entity_type_map = {}
for cate in entity_category:
    for item in entity_category[cate]:
        entity_type_map[item] = cate

def get_ref_data(data_path='./data/wikiref.json'):
    with open(data_path, 'r', encoding='utf-8') as reader:
        data = json.load(reader)
    refdata = {}
    for item in data:
        refdata[item['uid']] = item['summary']
    return refdata
refdata = get_ref_data(os.path.join(os.getenv("REFQA_DATA_DIR", "./data"), 'wikiref.json'))

spacy_ner    = spacy.load("en", disable=['parser', 'tagger'])
spacy_tagger = spacy.load("en", disable=['ner', 'parser'])
spacy_parser = spacy.load("en", disable=['ner', 'tagger'])
bene_parser = benepar.Parser("benepar_en2")

def get_new_question(context, answer, answer_start, summary, qtype):
    sentences = []
    for sent in summary:
        if answer in sent:
            sentences.append(sent)
    if len(sentences) == 0 or answer_start == -1:
        return None
    
    doc = spacy_parser(context)
    context_sent = None
    char_cnt = 0
    for sent_item in doc.sents:
        sent = sent_item.text
        if char_cnt <= answer_start < char_cnt + len(sent):
            context_sent = sent
            break
        else:
            char_cnt += len(sent)
            while char_cnt < len(context) and context[char_cnt] == ' ':
                char_cnt += 1

    if context_sent is None:
        return None

    c_tokens = []
    c_doc = spacy_tagger(context_sent)
    for token in c_doc:
        if not token.is_stop:
            c_tokens.append(token.lemma_)

    result = []
    for sent in sentences:
        sent_doc = spacy_tagger(sent)
        score = 0
        for token in sent_doc:
            if token.is_stop:
                continue
            if token.lemma_ in c_tokens:
                score += 1
        result.append([score, sent])
    result = sorted(result, key=lambda x: x[0])
    sentence = result[-1][1]
    cloze_text = None
    for clause in get_clause_v2(sentence, bene_parser):
        if answer in clause:
            cloze_text = clause.replace(answer, qtype, 1)
            break
    if cloze_text is None:
        return None

    new_question = identity_translate(reformulate_quesiton(cloze_text , spacy_parser, reform_version=1) )
    if new_question.startswith('Wh') or new_question.startswith('How'):
        return new_question
    else:
        return None

def get_answer_start(context, answer, orig_doc_start):
    begin_index = len(' '.join(context.split(' ')[:orig_doc_start]))
    answer_index = context.find(answer, begin_index)
    return answer_index

def generate(input_file, nbest_file, output_file, remove_em_answer=False, hard_em=False, score_lower_bound=0.5, debug=False):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    with open(nbest_file, "r", encoding='utf-8') as reader:
        nbest_data = json.load(reader)

    q_count = 0

    new_data = []
    for entry in (input_data if not debug else tqdm(input_data, desc='generate')):
        paras = []
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            qas = []
            for qa in paragraph['qas']:
                answer_text = qa['answers'][0]['text']
                qid = qa['id']
                cnt = 0
                for ans in (nbest_data[qid][:1] if hard_em else nbest_data[qid]):
                    if ans['probability'] < score_lower_bound:
                        continue
                    new_qa = copy.deepcopy(qa)
                    new_qa['id'] = qa['id']+'_%d'%cnt
                    ans['text'] = ans['text'].strip()

                    if debug:
                        new_qa['orig_question'] = qa['question']
                        new_qa['orig_answer'] = answer_text
                        new_qa['predict_answer'] = ans['text']
                        new_qa['summary'] = refdata[qid.split('_')[0]]

                    if (answer_text == ans['text']) or (ans['text'] in answer_text):
                        if remove_em_answer and answer_text == ans['text']:
                            continue
                        else:
                            qas.append(new_qa)
                    else:
                        new_qa['answers'][0]['text'] = ans['text']
                        new_qa['answers'][0]['answer_start'] = get_answer_start(context, ans['text'], ans['orig_doc_start'])
                        prev_qtype = qa['answers'][0]['type']
                        new_qa['question'] = get_new_question(context, ans['text'], new_qa['answers'][0]['answer_start'], refdata[qid.split('_')[0]], entity_type_map[prev_qtype])

                        if (new_qa['question'] is None) or (new_qa['answers'][0]['answer_start'] == -1):
                            continue
                        qas.append(new_qa)
                    cnt += 1

            if len(qas) == 0:
                continue
            q_count += len(qas)
            paragraph['qas'] = qas
            paras.append(paragraph)
        if len(paras) == 0:
            continue
        entry['paragraphs'] = paras
        new_data.append(entry)

    print('New Questions', q_count)

    json.dump({"version": "v2.0", 'data': new_data}, open(output_file, 'w', encoding='utf-8'), indent=4)

def generate2(input_file, nbest_file, output_file , hard_em=False, score_lower_bound=0.5, debug=False):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    with open(nbest_file, "r", encoding='utf-8') as reader:
        nbest_data = json.load(reader)

    q_count = 0
    em_qids = []

    new_data = []
    for entry in copy.deepcopy(input_data):
        paras = []
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            qas = []
            for qa in paragraph['qas']:
                answer_text = qa['answers'][0]['text']
                qid = qa['id']
                cnt = 0
                for ans in (nbest_data[qid][:1] if hard_em else nbest_data[qid]):
                    if ans['probability'] < score_lower_bound:
                        continue
                    new_qa = copy.deepcopy(qa)
                    new_qa['id'] = qa['id']+'_%d'%cnt
                    ans['text'] = ans['text'].strip()
                    if debug:
                        new_qa['orig_question'] = qa['question']
                        new_qa['orig_answer'] = answer_text
                        new_qa['predict_answer'] = ans['text']
                        new_qa['summary'] = refdata[qid.split('_')[0]]

                    if (answer_text == ans['text']) or (ans['text'] in answer_text):
                        if answer_text == ans['text']:
                            em_qids.append(qid)
                        else:
                            qas.append(new_qa)
                    else:
                        new_qa['answers'][0]['text'] = ans['text']
                        new_qa['answers'][0]['answer_start'] = get_answer_start(context, ans['text'], ans['orig_doc_start'])
                        prev_qtype = qa['answers'][0]['type']
                        new_qa['question'] = get_new_question(context, ans['text'], new_qa['answers'][0]['answer_start'], refdata[qid.split('_')[0]], entity_type_map[prev_qtype])

                        if (new_qa['question'] is None) or (new_qa['answers'][0]['answer_start'] == -1):
                            continue
                        qas.append(new_qa)
                    cnt += 1

            if len(qas) == 0:
                continue
            q_count += len(qas)
            paragraph['qas'] = qas
            paras.append(paragraph)
        if len(paras) == 0:
            continue
        entry['paragraphs'] = paras
        new_data.append(entry)

    random.shuffle(em_qids)
    em_qids = em_qids[:q_count]
    new_data += filter_data_given_qids(input_data, em_qids)
    q_count += len(em_qids)
    print('New Questions', q_count)

    json.dump({"version": "v2.0", 'data': new_data}, open(output_file, 'w', encoding='utf-8'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, type=str,
                        help="output_file")
    parser.add_argument("--input", default=None, type=str,
                        help="input_file")
    parser.add_argument("--nbest", default=None, type=str,
                        help="nbest_file")
    parser.add_argument("--generate_method", default=-1, type=int,
                        help="The method of generating new qa data.")
    parser.add_argument("--score_threshold", default=0.3, type=float,
                        help="The threshold of generating new qa data.")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed")
    args = parser.parse_args()
    random.seed(args.seed)

    if args.generate_method == 1:
        generate(args.input, args.nbest, args.output, remove_em_answer=True, score_lower_bound=args.score_threshold)
    elif args.generate_method == 2:
        generate2(args.input, args.nbest, args.output, score_lower_bound=args.score_threshold)
    elif args.generate_method == 3:
        generate(args.input, args.nbest, args.output, remove_em_answer=True, hard_em=True, score_lower_bound=args.score_threshold)
    elif args.generate_method == 4:
        generate2(args.input, args.nbest, args.output, hard_em=True, score_lower_bound=args.score_threshold)
    else:
        pass