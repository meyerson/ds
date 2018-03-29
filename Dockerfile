FROM jupyter/datascience-notebook

# sticking to python3 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY jupyter_notebook_config.py /opt/conda/etc/jupyter

