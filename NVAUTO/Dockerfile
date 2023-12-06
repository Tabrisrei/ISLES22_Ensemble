FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /opt/algorithm/ts \
    && chown algorithm:algorithm /opt/algorithm /input /output /opt/algorithm/ts

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt 

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm ./ts /opt/algorithm/ts

ENTRYPOINT python -m process $0 $@
