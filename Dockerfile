FROM continuumio/miniconda3:latest

WORKDIR /crm

COPY . .

RUN conda create --name crm_env python=3.11
RUN echo "source activate crm" > ~/.bashrc
ENV PATH /opt/conda/envs/crm/bin:$PATH

RUN conda env update --name crm --file environment.yaml --prune

CMD ["python", "train.py"]
