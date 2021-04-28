import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from shutil import copyfile

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    sub_dir = config["path"]["sub_dir_name"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    fixed_text_path = config["path"]["fixed_text_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    fixed_text_dict = dict()
    with open(fixed_text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            wav, fixed_text = line.split('|')[0], line.split('|')[1]
            fixed_text_dict[wav] = fixed_text.replace('\n', '')

    for sep_dir in tqdm(next(os.walk(in_dir))[1]):
        for clip_name in os.listdir(os.path.join(in_dir, sep_dir)):
            for file_name in os.listdir(os.path.join(in_dir, sep_dir, clip_name)):
                if file_name[-4:] != ".wav":
                    continue
                base_name = file_name[:-4]
                text_path = os.path.join(
                    in_dir, sep_dir, clip_name, "{}.txt".format(base_name)
                )
                wav_path = os.path.join(
                    in_dir, sep_dir, clip_name, "{}.wav".format(base_name)
                )
                if base_name in fixed_text_dict:
                    text = fixed_text_dict[base_name]
                else:
                    with open(text_path) as f:
                        text = f.readline().strip("\n")
                text = _clean_text(text, cleaners)

                os.makedirs(os.path.join(out_dir, sub_dir, clip_name), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, sub_dir, clip_name, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, sub_dir, clip_name, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)

    # Filelist
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as filelist:
        for line in tqdm(filelist.readlines()):
            wav = line.split('|')[0]
            if wav in fixed_text_dict:
                filelist_fixed.write("|".join([line.split("|")[0]] + [fixed_text_dict[wav]] + line.split("|")[2:]))
            else:
                filelist_fixed.write(line)
    filelist_fixed.close()

    # Speaker Info
    copyfile(f'{in_dir}/speaker_info.txt', f'{out_dir}/speaker_info.txt')