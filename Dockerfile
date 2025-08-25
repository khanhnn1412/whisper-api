# Base: Ubuntu 22.04 + CUDA 12.2 runtime
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set tzdata non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# --- Install Python 3.10.8 ---
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz && \
    tar xvf Python-3.10.8.tgz && \
    cd Python-3.10.8 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.10.8*

# Symlink python3.10.8 → python3 & pip3
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install deps
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app code
COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]