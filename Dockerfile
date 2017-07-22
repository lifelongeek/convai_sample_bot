FROM calee/parlai-gpu:env

WORKDIR /app

ADD . /app

RUN conda install -y cudnn

RUN cat cudapath >> ~/.bashrc

RUN . ~/.bashrc

RUN pip install cupy

EXPOSE 1990

CMD ["python", "run-demo-simple.py"]
