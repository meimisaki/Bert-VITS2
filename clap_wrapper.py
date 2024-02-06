import sys

import torch
from CLAP.msclap import CLAP
from config import config

models = dict()
LOCAL_PATH = "./CLAP/CLAP_weights_2023.pth"

def get_clap_audio_feature(audio_path, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = CLAP(LOCAL_PATH, version = '2023', device=device)
    with torch.no_grad():
        emb = models[device].get_audio_embeddings(audio_path)
    return emb.T


def get_clap_text_feature(text, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = CLAP(LOCAL_PATH, version = '2023', device=device)
    with torch.no_grad():
        emb = models[device].get_text_embeddings(text)
    return emb.T
