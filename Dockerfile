FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY ./ /opt/algorithm/

RUN mkdir -p /opt/algorithm /input /output /output_teams \
    &&groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm \
    &&chown -R algorithm:algorithm /opt/algorithm/ /input /output /output_teams

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    &&pip install -e SEALS/ \
    &&pip install -e FACTORIZER/model/factorizer/ \
    &&pip install -r requirements.txt

ENTRYPOINT ["bash", "Major_Voting_launcher.sh"]


