export REFQA_DATA_DIR=/root/data/refqa
export PYTORCH_PRETRAINED_BERT_CACHE=/root/pretrained_weights
export MAIN_MODEL_DIR=/root/model_outputs/best_main_model
export OUTPUT_DIR=/root/model_outputs/refqa_refine_model_output

cd ..

python multi_turn.py \
      --output_dir $OUTPUT_DIR \
      --model_dir $MAIN_MODEL_DIR \
      --predict_file $REFQA_DATA_DIR/dev-v1.1.json \
      --generate_method 2 \
      --score_threshold 0.15 \
      --threshold_rate 0.9 \
      --seed 17 \
      --fp16