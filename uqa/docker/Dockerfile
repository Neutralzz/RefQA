FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get update; apt-get install -y vim wget
RUN git clone -q https://github.com/NVIDIA/apex.git
RUN cd apex ; git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8 ;python setup.py install --user --cuda_ext --cpp_ext

RUN pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
RUN python -c "import nltk; nltk.download('punkt')"
RUN pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval

RUN mkdir ~/code; cd ~/code ; git clone https://github.com/microsoft/unilm.git
RUN cd ~/code/unilm/src ; pip install --user --editable .

RUN pip install --user cython spacy pytorch-transformers==1.2.0 tensorflow-gpu==1.13.1
RUN python -m spacy download en
RUN pip install --user benepar[gpu]

WORKDIR /workspace
RUN chmod -R a+w /workspace