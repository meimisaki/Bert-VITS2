# flake8: noqa: E402
import os
import logging
import re_matching
from tools.sentence import split_by_language

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
import librosa

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,
    cut_by_para, cut_by_sent, interval_between_para, interval_between_sent,
    reference_audio, emotion, style_text, style_weight,
):
    if style_text == "":
        style_text = None
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio = infer(
                p,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
    else:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            for idx, s in enumerate(sent_list):
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sent_list) - 1) and skip_end
                audio = infer(
                    s,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    style_text=style_text,
                    style_weight=style_weight,
                    skip_start=skip_start,
                    skip_end=skip_end,
                )
                audio_list_sent.append(audio)
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


def tts_fn(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,
    reference_audio, emotion, style_text, style_weight,
):
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = []
        for slice in re_matching.text_matching(text):
            _speaker = slice.pop()
            temp_contant = []
            temp_lang = []
            for lang, content in slice:
                if "|" in content:
                    temp = []
                    temp_ = []
                    for i in content.split("|"):
                        if i != "":
                            temp.append([i])
                            temp_.append([lang])
                        else:
                            temp.append([])
                            temp_.append([])
                    temp_contant += temp
                    temp_lang += temp_
                else:
                    if len(temp_contant) == 0:
                        temp_contant.append([])
                        temp_lang.append([])
                    temp_contant[-1].append(content)
                    temp_lang[-1].append(lang)
            for i, j in zip(temp_lang, temp_contant):
                result.append([*zip(i, j), _speaker])
        for i, one in enumerate(result):
            skip_start = i != 0
            skip_end = i != len(result) - 1
            _speaker = one.pop()
            idx = 0
            while idx < len(one):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    lang, content = one[idx]
                    temp_text = [content]
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(one):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(one) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        style_text,
                        style_weight,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        style_text,
                        style_weight,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)

def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    # audio = librosa.resample(audio, 44100, 48000)
    return sr, audio

def tts_dispatch(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,
    cut_by_para, cut_by_sent, interval_between_para, interval_between_sent,
    prompt_mode, reference_audio, emotion, style_text, style_weight,
):
    if style_text == "":
        style_text = None
    if prompt_mode == "Audio prompt":
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        else:
            reference_audio = load_audio(reference_audio)[1]
    else:
        reference_audio = None
    if cut_by_para:
        return tts_split(
            text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,
            cut_by_para, cut_by_sent, interval_between_para, interval_between_sent,
            reference_audio, emotion, style_text, style_weight,
        )
    else:
        return tts_fn(
            text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,
            reference_audio, emotion, style_text, style_weight,
        )

if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["JP"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.TextArea(
                        label="Text",
                        placeholder="Enter text here, currently only supports Japanese",
                    )
                    speaker = gr.Dropdown(
                        choices=speakers, value=speakers[0], label="Speaker"
                    )
                with gr.Group():
                    sdp_ratio = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.1, label="SDP Ratio"
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise"
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise_W"
                    )
                    length_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length"
                    )
                    language = gr.Dropdown(
                        choices=languages, value=languages[0], label="Language"
                    )
                syn_btn = gr.Button("Synthesize", variant="primary")
            with gr.Column():
                with gr.Accordion(label="About", open=False):
                    _ = gr.Markdown(value="""${about}""")
                with gr.Row():
                    with gr.Column():
                        prompt_mode = gr.State("Text prompt")
                        with gr.Tab(label="Text prompt") as text_tab:
                            text_prompt = gr.Textbox(
                                label="Emotional text",
                                placeholder="Describe your desired emotion in a few words, e.g: Happy",
                                value="Neutral",
                            )
                        with gr.Tab(label="Audio prompt") as audio_tab:
                            audio_prompt = gr.Audio(
                                label="Reference audio", type="filepath"
                            )
                            audio_prompt.upload(
                                lambda x: load_audio(x),
                                inputs=[audio_prompt],
                                outputs=[audio_prompt],
                            )
                        text_tab.select(
                            lambda: "Text prompt", inputs=[], outputs=[prompt_mode]
                        )
                        audio_tab.select(
                            lambda: "Audio prompt", inputs=[], outputs=[prompt_mode]
                        )
                with gr.Group(visible=False):
                    style_text = gr.Textbox(
                        label="Stylization text",
                        placeholder="Fine-grained control of voice emotions",
                    )
                    style_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        step=0.1,
                        label="Stylization strength",
                    )
                with gr.Group():
                    opt_cut_by_sent = gr.Checkbox(
                        label="Split text by sentences"
                    )
                    interval_between_sent = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=0.2,
                        step=0.1,
                        label="Pause time between sentences (in seconds)",
                    )
                    opt_cut_by_para = gr.Checkbox(
                        label="Split text by paragraphs"
                    )
                    interval_between_para = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=1,
                        step=0.1,
                        label="Pause time between paragraphs (in seconds)",
                    )
                    opt_cut_by_sent.input(
                        lambda x, y: x or y,
                        inputs=[opt_cut_by_para, opt_cut_by_sent],
                        outputs=[opt_cut_by_para]
                    )
                    opt_cut_by_para.input(
                        lambda x, y: x and y,
                        inputs=[opt_cut_by_para, opt_cut_by_sent],
                        outputs=[opt_cut_by_sent]
                    )
                with gr.Group():
                    text_output = gr.Textbox(label="Info")
                    audio_output = gr.Audio(label="Output")

        syn_btn.click(
            tts_dispatch,
            inputs=[
                text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,
                opt_cut_by_para, opt_cut_by_sent, interval_between_para, interval_between_sent,
                prompt_mode, audio_prompt, text_prompt, style_text, style_weight,
            ],
            outputs=[text_output, audio_output],
        )

    print("推理页面已开启!")
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
