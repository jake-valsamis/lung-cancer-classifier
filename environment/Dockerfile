# hash:sha256:adb5d8f9c0c521af8969bbc69f69b004052034a5946053c27ac983b206fe6b35
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN mamba install -y \
        jupyterlab==4.2.5 \
        mlflow==2.17.0 \
        pandas==2.2.3 \
        scikit-learn==1.5.2 \
        xgboost==2.1.1 \
    && mamba clean -ya
