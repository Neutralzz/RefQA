# -*- coding: utf-8 -*-
import sys, os, json, time
import numpy as np
from tqdm import tqdm
import random
import logging
import argparse
import subprocess

from evaluate import evaluate

logger = logging.getLogger('multi-turn')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def get_nbest_file(model_dir, dev_file, params):
    command = 'python run_squad.py --model_type bert \
            --model_name_or_path bert-large-uncased-whole-word-masking \
            --do_eval --do_lower_case --train_file %s \
            --predict_file %s --max_seq_length 384  --doc_stride 128 --output_dir %s \
            --per_gpu_eval_batch_size=12 --eval_prefix dev' % (params.predict_file, dev_file, model_dir)
    if params.fp16:
        command += ' --fp16'
    nbest_file = os.path.join(model_dir, 'nbest_predictions_dev.json')
    if params.debug and os.path.isfile(nbest_file):
        logger.info('%s already existed and we use it.'%nbest_file)
    else:
        logger.info('Generating nbest file...')
        subprocess.Popen(command, shell=True).wait()
    
    if not os.path.isfile(nbest_file):
        logger.error('Nbest file %s is not found.'%nbest_file)
        exit()
    logger.info('Got nbest file %s'%nbest_file)
    return nbest_file

def get_new_train_file(dev_file, nbest_file, model_dir, params):
    new_train_file = os.path.join(model_dir, 'train_data_for_next_turn.json')
    command = 'python generate_new_qadata.py --input %s --nbest %s --output %s \
            --generate_method %d --score_threshold %.4f --seed %d'%(dev_file, nbest_file, new_train_file, params.generate_method, params.score_threshold, params.seed)
    subprocess.Popen(command, shell=True).wait()
    if not os.path.isfile(new_train_file):
        logger.error('New train file %s is not found.'%new_train_file)
        exit()
    logger.info('Got new train file %s'%new_train_file)
    return new_train_file


def do_evaluate(dataset_file, prediction_file):
    with open(dataset_file) as df:
        dataset_json = json.load(df)
        dataset = dataset_json['data']
    with open(prediction_file) as pf:
        predictions = json.load(pf)
    return evaluate(dataset, predictions)

def train_model(train_file, model_dir, output_dir, params):
    command = 'python -m torch.distributed.launch --nproc_per_node=4 run_squad.py \
        --model_type bert  --model_name_or_path %s --do_train  --do_eval  --do_lower_case \
        --train_file %s --predict_file %s \
        --learning_rate 3e-5  --num_train_epochs 1.0  --max_seq_length 384  --doc_stride 128 \
        --output_dir %s  --per_gpu_eval_batch_size=6  --per_gpu_train_batch_size=6 --seed %d \
        --logging_steps 1000  --save_steps 1000 --eval_all_checkpoints \
        --overwrite_output_dir --overwrite_cache'%(model_dir, train_file, params.predict_file, output_dir, params.seed)
    if params.fp16:
        command += ' --fp16'
    subprocess.Popen(command, shell=True).wait()

    # select best model for next turn
    new_model_dir = output_dir
    score = do_evaluate(params.predict_file, os.path.join(output_dir, 'predictions_.json'))['f1']

    for filename in os.listdir(output_dir):
        if (not filename.startswith('predictions_')) or (filename == 'predictions_.json'):
            continue
        new_score = do_evaluate(params.predict_file, os.path.join(output_dir, filename))['f1']
        if new_score > score:
            score = new_score
            ckpt = filename.replace('.json', '').replace('predictions_', 'checkpoint-')
            new_model_dir = os.path.join(output_dir, ckpt)
            subprocess.Popen('cp %s/vocab.txt  %s'%(output_dir, new_model_dir), shell=True).wait()
            subprocess.Popen('cp %s/special_tokens_map.json  %s'%(output_dir, new_model_dir), shell=True).wait()
            subprocess.Popen('cp %s/added_tokens.json  %s'%(output_dir, new_model_dir), shell=True).wait()
            subprocess.Popen('cp %s/%s  %s/predictions_.json'%(output_dir, filename, new_model_dir), shell=True).wait()
            

    return new_model_dir, score

    

def main(params):
    dev_data_name = os.path.join(args.refine_data_dir, 'uqa_train_refine_%d.json')

    model_dir = os.path.join(params.output_dir, 'init')
    if not os.path.exists(model_dir):
        subprocess.Popen('mkdir -p %s'%model_dir, shell=True).wait()
    logger.info('Copy model from %s to %s.'%(params.model_dir, model_dir))
    subprocess.Popen('cp %s/vocab.txt %s'%(params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/special_tokens_map.json %s'%(params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/added_tokens.json %s'%(params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/config.json %s'%(params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/training_args.bin %s'%(params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/predictions_.json %s'%(params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/pytorch_model.bin %s'%(params.model_dir, model_dir), shell=True).wait()

    if params.debug:
        subprocess.Popen('cp %s/nbest_predictions_6_no_train_eval2.json %s/nbest_predictions_dev.json'%(params.model_dir, model_dir), shell=True).wait()

    if os.path.exists(os.path.join(model_dir, 'predictions_.json')):
        current_score = do_evaluate(params.predict_file, os.path.join(model_dir, 'predictions_.json'))['f1']
    else:
        current_score = 0.0

    order = [1, 3, 2, 4, 5, 0]
    if params.debug:
        order = [6] + order

    for step, idx in enumerate(order):
        logger.info('-'*80)
        logger.info('Prepare for turn_%d / Current f1 %.2f/ Current model %s'%(step, current_score, model_dir))
        dev_file = dev_data_name % idx
        output_dir = os.path.join(params.output_dir, 'turn_%d'%step)
        if not os.path.exists(output_dir):
            subprocess.Popen('mkdir -p %s'%output_dir, shell=True).wait()

        nbest_file = get_nbest_file(model_dir, dev_file, params)
        new_train_file = get_new_train_file(dev_file, nbest_file, model_dir, params)

        new_model_dir, new_score = train_model(new_train_file, model_dir, output_dir, params)

        if new_score > current_score:
            model_dir = new_model_dir
            current_score = new_score
            logger.info('Find better model %s and f1 is %.2f'%(model_dir, current_score))

        params.score_threshold = params.score_threshold * params.threshold_rate




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine_data_dir", default=None, type=str, required=True,
                        help="RefQA data for refining.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The init model directory.")
    parser.add_argument("--predict_file", default='dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--generate_method", default=1, type=int,
                        help="The method of generating new qa data.")
    parser.add_argument("--score_threshold", default=0.3, type=float,
                        help="The threshold of generating new qa data.")
    parser.add_argument("--threshold_rate", default=1.0, type=float,
                        help="The change rate of the threshold")
    parser.add_argument("--seed", default=42, type=int,
                        help="seed")
    parser.add_argument("--fp16", action='store_true',
                        help="fp16 training")
    parser.add_argument("--debug", action='store_true',
                        help="debug training")    
    args = parser.parse_args()
    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    main(args)
