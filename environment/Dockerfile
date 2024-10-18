# hash:sha256:ad2bd91ea1488269f26f030bb42cf2018f26395c7bb474a1981c394ed2c3641d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN mamba install -y \
        jupyterlab==4.2.5 \
    && mamba clean -ya

RUN pip install -U --no-cache-dir \
    mlflow==2.17.0 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    xgboost==2.1.1
