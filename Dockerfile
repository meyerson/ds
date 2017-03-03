FROM python:2.7

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/opt/app/src \
    C_FORCE_ROOT="true"

# Volumes
VOLUME ['/opt/app', '/opt/data']

# get up to date and install packages

RUN apt-get update --fix-missing && \
    apt-get install -y \
        python2.7-dev \
        libxml2-dev \
        libxslt1-dev \
        sqlite3 \
        libsqlite3-dev \
        libhdf5-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libsnappy-dev \
        graphviz

# Makes the build faster as it caches the install as a layer
RUN pip install python-dateutil==2.2
RUN pip install Cython==0.23.4
RUN pip install numpy==1.10.4
RUN pip install scipy==0.17
RUN pip install jupyter
RUN pip install cython
RUN pip install supervisor
# RUN pip install cogent



RUN mkdir -p /opt/app/src

COPY requirements.txt /opt/app/src/requirements.txt
RUN pip install --no-binary :all -r /opt/app/src/requirements.txt


RUN mkdir -p /var/log/luigid
COPY start.sh /opt/app/start.sh
COPY conf/* /opt/app/conf/
COPY start.sh /opt/app/start.sh

ENV USER_PORT=8902
ENV LUIGI_PORT=8082
ENV PYTHONPATH=/opt/app/src

EXPOSE 8902 8082

WORKDIR /opt/app

ENTRYPOINT ["/opt/app/start.sh"]
