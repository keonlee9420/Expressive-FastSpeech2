import os
import re
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

_square_brackets_re = re.compile(r"\[[\w\d\s]+\]")
_inv_square_brackets_re = re.compile(r"(.*?)\](.+?)\[(.*)")


def get_sorted_items(items):
    # sort by key
    return sorted(items, key=lambda x:x[0])


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    sub_dir = config["path"]["sub_dir_name"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    fixed_text_path = config["path"]["fixed_text_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    os.makedirs(os.path.join(out_dir), exist_ok=True)
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    speaker_info, speaker_done = dict(), set()

    fixed_text_dict = dict()
    with open(fixed_text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            wav, fixed_text = line.split('|')[0], line.split('|')[1]
            fixed_text_dict[wav] = fixed_text.replace('\n', '')

    for sep_dir in tqdm(next(os.walk(in_dir))[1]):
        if sub_dir[:-1] not in sep_dir.lower():
            continue
        for wav_dir in tqdm((next(os.walk(os.path.join(in_dir, sep_dir, "sentences", "wav")))[1])):

            # Build Text Dict
            text_dict = dict()
            text_raw_path = os.path.join(
                in_dir, sep_dir, "dialog", "transcriptions", "{}.txt".format(wav_dir)
            )
            with open(text_raw_path) as f:
                for line in f.readlines():
                    base_name = line.split("[")[0].strip()
                    transcript = line.split("]:")[-1].strip()
                    text_dict[base_name] = transcript
                    
            # Build Emotion Dict
            emo_dict = dict()
            emo_raw_path = os.path.join(
                in_dir, sep_dir, "dialog", "EmoEvaluation", "{}.txt".format(wav_dir)
            )
            with open(emo_raw_path) as f:
                for line in f.readlines()[1:]:
                    if "[" not in line or "%" in line:
                        continue
                    m = _inv_square_brackets_re.match(" ".join(line.split()))
                    base_name, emo_gt = m.group(2).strip().split(" ")
                    valence, arousal = m.group(3).split(",")[0].strip(), m.group(3).split(",")[1].strip()
                    emo_dict[base_name] = {
                        "e": emo_gt,
                        "a": arousal,
                        "v": valence,
                    }

            for file_name in os.listdir(os.path.join(in_dir, sep_dir, "sentences", "wav", wav_dir)):
                if file_name[0] == "." or file_name[-4:] != ".wav":
                    continue
                base_name = file_name[:-4]
                if len(base_name.split("_")) == 3:
                    spk_id, dialog_type, turn = base_name.split("_")
                elif len(base_name.split("_")) == 4:
                    spk_id, dialog_type, turn = base_name.split("_")[0], "_".join(base_name.split("_")[1:3]), base_name.split("_")[3]
                base_name_new = "_".join([turn, spk_id, dialog_type])

                if spk_id not in speaker_done:
                    speaker_info[spk_id] = {
                        'gender': spk_id[-1]
                    }
                    speaker_done.add(spk_id)

                wav_path = os.path.join(
                    in_dir, sep_dir, "sentences", "wav", wav_dir, "{}.wav".format(base_name)
                )
                if base_name in fixed_text_dict:
                    text = fixed_text_dict[base_name]
                else:
                    text = text_dict[base_name]
                text = re.sub(_square_brackets_re, "", text)
                text = ' '.join(text.split())
                text = _clean_text(text, cleaners)

                os.makedirs(os.path.join(out_dir, sub_dir, wav_dir), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, sub_dir, wav_dir, "{}.wav".format(base_name_new)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, sub_dir, wav_dir, "{}.lab".format(base_name_new)),
                    "w",
                ) as f1:
                    f1.write(text)
                
                # Filelist
                emo_ = emo_dict[base_name]
                emotion, arousal, valence = emo_["e"], emo_["a"], emo_["v"]
                filelist_fixed.write("|".join([base_name_new, text, spk_id, emotion, arousal, valence]) + "\n")
    filelist_fixed.close()

    # Save Speaker Info
    with open(f'{out_dir}/speaker_info.txt', 'w', encoding='utf-8') as f:
        for spk_id, spk_info in get_sorted_items(speaker_info.items()):
            gender = spk_info['gender']
            f.write(f'{spk_id}|{gender}\n')