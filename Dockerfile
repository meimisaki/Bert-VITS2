FROM python:3.10.12

# Install Git LFS
RUN apt-get update -y
RUN apt-get install -y git-lfs
RUN git lfs install

# Switch to User
RUN useradd -m -u 1000 user
USER user

# Setup Environment Variables
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Init Working Directory
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Install Python Requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

# Download Bert, CLAP and WavLM Models
RUN rm -rf bert/deberta-v2-large-japanese-char-wwm
RUN git clone https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm bert/deberta-v2-large-japanese-char-wwm --depth=1
RUN cd bert/deberta-v2-large-japanese-char-wwm && git lfs pull
RUN rm -rf emotional/clap-htsat-fused
RUN git clone https://huggingface.co/laion/clap-htsat-fused emotional/clap-htsat-fused --depth=1
RUN cd emotional/clap-htsat-fused && git lfs pull
RUN rm -rf slm/wavlm-base-plus
RUN git clone https://huggingface.co/microsoft/wavlm-base-plus slm/wavlm-base-plus --depth=1
RUN cd slm/wavlm-base-plus && git lfs pull

# Start Web Server
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "webui.py"]
