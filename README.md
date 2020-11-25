<h2 align="center">
    <p>Harvesting and Refining Question-Answer Pairs for Unsupervised QA</p>
</h1>

This repo contains the data, codes and models for the ACL2020 paper ["Harvesting and Refining Question-Answer Pairs for Unsupervised QA"](https://arxiv.org/abs/2005.02925).

In this work, we introduce two approaches to improve unsupervised QA. First, we harvest lexically and syntactically divergent questions from Wikipedia to automatically construct a corpus of question-answer pairs (named as RefQA). Second, we take advantage of the QA model to extract more appropriate answers, which iteratively refines data over RefQA. We conduct experiments on SQuAD 1.1, and NewsQA by fine-tuning BERT without access to manually annotated data. Our approach outperforms previous unsupervised approaches by a large margin and is competitive with early supervised models.

## Environment

### With Docker

The recommended way to run the code is using docker under Linux. The Dockerfile is in `uqa/docker/Dockerfile`.

### With Pip

First you need to install PyTorch 1.1.0. Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally). 

Then, you can clone this repo and install dependencies by `uqa/scripts/install_tools.sh`:

```bash
git clone -q https://github.com/NVIDIA/apex.git
cd apex ; git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python3 setup.py install --user --cuda_ext --cpp_ext

pip install --user cython tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
python3 -c "import nltk; nltk.download('punkt')"
pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval

pip install --user spacy==2.2.0 pytorch-transformers==1.2.0 tensorflow-gpu==1.13.1
python3 -m spacy download en
pip install --user benepar[gpu]
```

The mixed-precision training code requires the specific version of [NVIDIA/apex](https://github.com/NVIDIA/apex/tree/1603407bf49c7fc3da74fceb6a6c7b47fece2ef8), which only supports pytorch<1.2.0.

## Data and Models

The format of our generated data is SQuAD-like. The data can be downloaded from [here](https://drive.google.com/open?id=18o8EjlCcimvuF0HYe8sHSu6epTqDwvp_).

The links to the trained models:
- [refqa-main](https://drive.google.com/open?id=1r2jgFSGtXBRTAeFzGzAwQ_BG4_Bi8v7f): The trained model using 300k RefQA examples;
- [refqa-refine](https://drive.google.com/open?id=1wiAV7sYQFhXVNCuVK8kk9S114_z7Rjwc): The trained model by our refining process.

## Constructing RefQA

In our released data, the `wikiref.json` file (our raw data) contains the Wikipedia statements and corresponding cited documents (the `summary` and `document` key for each item).

You can convert the raw data to our RefQA by the following script:

```bash
export REFQA_DATA_DIR=/{path_to_refqa_data}/
 
python3 wikiref_process.py \
        --input_file wikiref.json \
        --output_file cloze_clause_wikiref_data.json
python3 cloze2natural.py \
        --input_file cloze_clause_wikiref_data.json \
        --output_file refqa.json
```

Note: Please make sure that the file `wikiref.json` is in the directory `$REFQA_DATA_DIR`.

Then, for the following refining process, you should split your generated data to several parts, such as a main data to train an initial QA model and other parts to do refining process.

## Training and Refining

Before running on RefQA, you should download/move the [data](#data-and-models) and the SQuAD 1.1 dev file `dev-v1.1.json` to the directory `$REFQA_DATA_DIR`.

We train our QA model using distributed and mixed-precision training on 4 P100 GPUs.

### Training the initial QA model

You can fine-tune BERT-Large (WWM) on 300k RefQA examples and achieve a F1 > 65 on SQuAD 1.1 dev set.

```bash
export REFQA_DATA_DIR=/{path_to_refqa_data}/
export OUTPUT_DIR=/{path_to_main_output}/
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=4 run_squad.py \
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
        --save_steps 1000
```

### Refining RefQA data iteratively

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1r2jgFSGtXBRTAeFzGzAwQ_BG4_Bi8v7f)) used for refining process. The refining process is conducted as follows:

```bash
export REFQA_DATA_DIR=/{path_to_refqa_data}/
export MAIN_MODEL_DIR=/{path_to_previous_fine-tuned_model}/
export OUTPUT_DIR=/{path_to_refine_output}/
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 multi_turn.py \
      --refine_data_dir $REFQA_DATA_DIR \
      --output_dir $OUTPUT_DIR \
      --model_dir $MAIN_MODEL_DIR \
      --predict_file $REFQA_DATA_DIR/dev-v1.1.json \
      --generate_method 2 \
      --score_threshold 0.15 \
      --threshold_rate 0.9 \
      --seed 17 \
      --fp16
```

The `multi_turn.py` provides the following command line arguments:

```
positional arguments:
    --refine_data_dir   The directory of RefQA data for refining
    --model_dir         The directory of the init checkpoint
    --output_dir        The output directory
    --predict_file      SQuAD or other json for predictions. E.g., dev-v1.1.json

optional arguments:
    --generate_method   {1|2} The method of generating data for next training,
                        1 is using refined data only, 2 is merging refined data with filtered data (1:1 ratio)
    --score_threshold   The threshold for filtering predicted answers
    --threshold_rate    The decay factor for the above threshold
    --seed              Random seed for initialization
    --fp16              Whether to use 16-bit (mixed) precision (through NVIDIA apex)
```


## Citation
If you find this repo useful in your research, you can cite the following paper:
```
@inproceedings{li2020refqa,
    title = "Harvesting and Refining Question-Answer Pairs for Unsupervised {QA}",
    author = "Li, Zhongli  and
      Wang, Wenhui  and
      Dong, Li  and
      Wei, Furu  and
      Xu, Ke",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.600",
    doi = "10.18653/v1/2020.acl-main.600",
    pages = "6719--6728"
}
```

## Acknowledgment

Our code is based on [pytorch-transformers 1.2.0](https://github.com/huggingface/transformers/tree/1.2.0). We thank the authors for their wonderful open-source efforts.

