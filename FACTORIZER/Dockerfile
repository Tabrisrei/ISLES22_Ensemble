FROM python:3.9-slim



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm model /opt/algorithm/model
RUN python -m pip install --user model/factorizer/

COPY --chown=algorithm:algorithm predict.py /opt/algorithm/

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
