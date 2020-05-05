# -*- coding: utf-8 -*-
import sys, os, json, time
from tqdm import tqdm
import random
from collections import Counter
from nltk import word_tokenize
import numpy as np
from data_utils import reformulate_quesiton
import spacy
import copy

mask2wh = {
    'PERSONNORPORG' : 'Who',
    'PLACE' : 'Where',
    'THING' : 'What',
    'TEMPORAL': 'When',
    'NUMERIC' : ['How many','How much']
}
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
data_dir = os.getenv("REFQA_DATA_DIR", "./data")

def identity_translate(cloze_question):
    if 'NUMERIC' in cloze_question:
        return cloze_question.replace('NUMERIC', mask2wh['NUMERIC'][int(2*random.random())])
    else:
        for mask in mask2wh:
            if mask in cloze_question:
                return cloze_question.replace(mask, mask2wh[mask])
        raise Exception('\'{}\' should have one specific masked tag.'.format(cloze_question))

def word_shuffle(tokens, word_shuffle_param):
    length = len(tokens)
    noise = np.random.uniform(0, word_shuffle_param, size=(length ) )
    word_idx = np.array([1.0*i for i in range(length)])

    scores = word_idx + noise
    scores += 1e-6 * np.arange(length)
    permutation = scores.argsort()
    new_s = [ tokens[idx] for idx in permutation ]
    return new_s

def word_dropout(tokens, word_dropout_param):
    length = len(tokens)
    if word_dropout_param == 0:
        return tokens
    assert 0 < word_dropout_param < 1

    keep = np.random.rand(length) >= word_dropout_param
    #if length:
    #    keep[0] = 1
    new_s =  [ w for j, w in enumerate(tokens) if keep[j] ]
    return new_s

def word_mask(tokens, word_mask_param, mask_str='[MASK]'):
    length = len(tokens)
    if word_mask_param == 0:
        return tokens
    assert 0 < word_mask_param < 1

    keep = np.random.rand(length) >= word_mask_param
    #if length:
    #    keep[0] = 1
    new_s =  [ w if keep[j] else mask_str  for j, w in enumerate(tokens)]
    return new_s

def noisy_clozes_translate(cloze_question, params=[2, 0.2, 0.1]):
    wh = None
    for mask in mask2wh:
        if mask in cloze_question:
            cloze_question = cloze_question.replace(mask,'')
            wh = mask2wh[mask]
            break
    if isinstance(wh , list):
        wh = wh[int(2*random.random())]

    tokens = word_tokenize(cloze_question)
    tokens = word_shuffle(tokens, params[0])
    tokens = word_dropout(tokens, params[1])
    tokens = word_mask(tokens, params[2])
    return wh+' '+(' '.join(tokens))

def cloze_to_natural_questions(input_data, method):

    natural_data = []
    q_count = 0

    parser = spacy.load("en", disable=['ner', 'tagger'])

    for entry in tqdm(input_data, desc="cloze"):
        parags = []
        for paragraph in entry['paragraphs']:
            qas = []
            for qa in paragraph['qas']:
                qa['question'] = qa['question'].replace('PERSON/NORP/ORG', 'PERSONNORPORG')
                try:
                    if method == 0:
                        qa['question'] = identity_translate(qa['question'])
                    elif method == 1:
                        qa['question'] = noisy_clozes_translate(qa['question'])
                    elif method == 2:
                        qa['question'] = identity_translate(reformulate_quesiton(qa['question'], parser, reform_version=1) )
                    else:
                        raise NotImplementedError()    
                except Exception as e:
                    print(qa['question'])
                    print(repr(e))
                    continue

                qas.append(qa)
            paragraph["qas"] = qas
            parags.append(paragraph)
            q_count += len(qas)
        entry["paragraphs"] = parags
        natural_data.append(entry)    
        #if q_count > 10:
        #    break

    print('Questions Number', q_count)
    return {"version": "v2.0", 'data': natural_data}


def filter_data_given_qids(input_data_, qids):
    input_data = copy.deepcopy(input_data_)
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

def main(input_file, output_file, method):
    input_file = os.path.join(data_dir, input_file)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    natural_data = cloze_to_natural_questions(input_data, method)
    json.dump(natural_data, open(os.path.join(data_dir, output_file) , 'w', encoding='utf-8'), indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--method", default=2, type=int)
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.method)