git clone -q https://github.com/NVIDIA/apex.git
cd apex ; git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --user --cuda_ext --cpp_ext

pip install --user cython tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
python -c "import nltk; nltk.download('punkt')"
pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval

pip install --user spacy==2.2.0 pytorch-transformers==1.2.0 tensorflow-gpu==1.13.1
python -m spacy download en
pip install --user benepar[gpu]