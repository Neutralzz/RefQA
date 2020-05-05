export REFQA_DATA_DIR=/root/data/refqa
export PYTORCH_PRETRAINED_BERT_CACHE=/root/pretrained_weights
export OUTPUT_DIR=/root/model_outputs/refqa_main_model_output

cd ../ 
 
python -m torch.distributed.launch --nproc_per_node=4 run_squad.py \
        --model_type bert \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_train \
        --do_eval \
        --do_lower_case \
        --train_file $REFQA_DATA_DIR/uqa_train_main.json \
        --predict_file $REFQA_DATA_DIR/dev-v1.1.json \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir $OUTPUT_DIR \
        --per_gpu_train_batch_size=6 \
        --per_gpu_eval_batch_size=4 \
        --seed 42 \
        --fp16 \
        --overwrite_output_dir \
        --logging_steps 1000 \
        --save_steps 1000 ;
