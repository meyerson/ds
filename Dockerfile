FROM jupyter/datascience-notebook

# Env variables
RUN pip install luigi
COPY jupyter_notebook_config.py /opt/conda/etc/jupyter

