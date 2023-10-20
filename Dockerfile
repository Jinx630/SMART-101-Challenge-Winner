FROM reg.docker.alibaba-inc.com/dsw/pytorch:1.12-gpu-py39-cu113-ubuntu20.04

RUN apt-get -y update

COPY requirements.txt requirements.txt

COPY lavis  ./lavis
COPY llama  ./llama
COPY projects  ./projects
COPY run_scripts  ./run_scripts
COPY tests  ./tests
COPY yolov7  ./yolov7
COPY ckpts  ./ckpts

COPY 01-llama_questype.py  ./01-llama_questype.py
COPY 03-create-test.py  ./03-create-test.py
COPY evaluate.py  evaluate.py
COPY run.sh  ./run.sh
COPY setup.py  setup.py
COPY tokenizer_checklist.chk  ./tokenizer_checklist.chk
COPY tokenizer.model  ./tokenizer.model
COPY train.py  train.py

RUN pip3 install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple


CMD ["bash", "run.sh"]
