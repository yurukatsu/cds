ARG python_image="python:3.10-slim"
# pythonイメージをダウンロード
FROM ${python_image}

ARG work_dir="/root/cds/"
# コンテナにアクセスしたときのデフォルトディレクトリ
WORKDIR ${work_dir}

# curlインストール
RUN apt-get update && apt-get install -y &&\
    apt-get install curl -y &&\
    apt-get clean

# poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python3 -
# パスを通す
ENV PATH $PATH:/root/.local/bin
# 仮想関係を作成しないように設定
RUN poetry config virtualenvs.create false