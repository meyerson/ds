FROM jupyter/datascience-notebook

# sticking to python3 
RUN pip install luigi
RUN pip install sklearn-pandas
COPY jupyter_notebook_config.py /opt/conda/etc/jupyter

