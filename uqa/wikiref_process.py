# -*- coding: utf-8 -*-
import sys, os, json, time
from tqdm import tqdm
from multiprocessing import Process

import spacy
import benepar
import nltk

data_dir = os.getenv("REFQA_DATA_DIR", "./data")

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
#print(entity_type_map)

def search_sbar_from_tree(tree_node):
    if not isinstance(tree_node, nltk.Tree):
        return []
    clause = []
    for i in range(len(tree_node)):
        clause += search_sbar_from_tree(tree_node[i])
    #if tree_node.label() == 'SBAR':
    if tree_node.label() == 'S' or tree_node.label() == 'SBAR':
        clause.append(tree_node)
    return clause

def get_clause_v2(sentence, predictor):
    start = time.time()
    parsing_tree = predictor.parse(sentence)
    sbar = search_sbar_from_tree(parsing_tree)[:-1]
    sbar_text = [' '.join(item.leaves()) for item in sbar]

    result = []
    for node in sbar:
        if node.label() == 'S':
            item = ' '.join(node.leaves())
            if len(item.split()) <= 5:
                continue
            result.append(item)

    result = sorted(result, key=lambda x: len(x))
    
    result2 = []
    sentence = ' '.join(parsing_tree.leaves())
    clauses = sentence.split(',')
    for i in range(len(clauses)):
        item, p = clauses[i], i+1
        while len(item.split()) < 10 and p < len(clauses):
            item = ','.join([item, clauses[p]])
            p += 1
        result2.append(item.strip())

    result2 = sorted(result2, key=lambda x: len(x))
    return result + result2

def get_answer_start(answer, question, sentences, tagger):
    q_tokens = []
    q_doc = tagger(question)
    for token in q_doc:
        if not token.is_stop:
            q_tokens.append(token.lemma_)

    result = []
    for sent in sentences:
        if sent.find(answer) == -1:
            continue
        sent_doc = tagger(sent)
        score = 0
        for token in sent_doc:
            if token.is_stop:
                continue
            if token.lemma_ in q_tokens:
                score += 1
        result.append([score, sent])
    if len(result) == 0:
        return -1
    else:
        result = sorted(result, key=lambda x: x[0])
        res_sent = result[-1][1]

        answer_start = ' '.join(sentences).find(res_sent) + res_sent.find(answer)
        return answer_start


def get_cloze_data(input_data, clause_extract=False, proc = None):
    if clause_extract:
        parser = benepar.Parser("benepar_en2")

    ner = spacy.load("en", disable=['parser', 'tagger'])
    tagger = spacy.load("en", disable=['parser', 'ner'])

    cloze_data = []

    q_count = 0
    c_count = 0

    for item in tqdm(input_data, desc="cloze"):
        entry = {}
        entry['title'] = item["document"][0]
        paragraph = {}
        paragraph["context"] = ' '.join(item["document"])

        qas = []

        for sent in item['summary']:
            sent_doc = ner(sent)

            if clause_extract:
                try:
                    clause = get_clause_v2(sent, parser)
                except Exception as e:
                    continue
            
            for ent in sent_doc.ents:
                answer = ent.text

                question = None
                if clause_extract:
                    for each in clause:
                        if each.find(answer) != -1:
                            question = each.replace(answer, entity_type_map[ent.label_], 1)
                            break
                else:
                    question = sent[:ent.start_char] + \
                            sent[ent.start_char:].replace(answer,entity_type_map[ent.label_], 1)
                if not question:
                    continue


                answer_start = get_answer_start(answer, question, item['document'], tagger)
                if answer_start == -1:
                    continue

                qas.append({
                            "question": question,
                            "id": "%s_%d"%(item['uid'], q_count) ,
                            "is_impossible": False,
                            "answers": [
                                {
                                "answer_start": answer_start,
                                "text": answer,
                                "type": ent.label_
                                }
                            ],
                            "plausible_answers": []
                })
                q_count += 1

        paragraph['qas'] = qas
        entry['paragraphs'] = [paragraph]
        
        cloze_data.append(entry)
        #if q_count > 10:
        #    break
        c_count += 1
        if c_count % 2000 == 0:
            print(proc, 'processing %d/%d ...'%(c_count, len(input_data) ))


    if proc is not None:
        json.dump(cloze_data, open(os.path.join(data_dir, 'tmp_store_%d.json'%proc), 'w', encoding='utf-8'))

    print('Questions Number', q_count)    
    return {"version": "v2.0", 'data': cloze_data}


def main(args):
    input_file = os.path.join(data_dir, args.input_file)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    cloze_clause_data = get_cloze_data(input_data, clause_extract=True)
    json.dump(cloze_clause_data, open(os.path.join(data_dir, args.output_file),
                "w", encoding='utf-8'), indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="wikiref.json", type=str)
    parser.add_argument("--output_file", default="cloze_clause_wikiref_data.json", type=str)
    args = parser.parse_args()
    main(args)


