import re
import argparse
import yaml
import os
import shutil
import json
import librosa
import soundfile
from glob import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from text import _clean_text
from text.korean import tokenize, normalize_nonchar


def write_text(txt_path, text):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_sorted_items(items):
    # sort by key
    return sorted(items, key=lambda x:int(x[0]))


def get_emotion(emo_dict):
    e, a, v = 0, 0, 0
    if 'emotion' in emo_dict:
        e = emo_dict['emotion']
        a = emo_dict['arousal']
        v = emo_dict['valence']
    return e, a, v


def pad_spk_id(speaker_id):
    return 'p{}'.format("0"*(3-len(speaker_id))+speaker_id)


def create_dataset(preprocess_config):
    """
    See https://github.com/Kyumin-Park/aihub_multimodal_speech
    """
    in_dir = preprocess_config["path"]["corpus_path"]
    audio_dir = os.path.join(os.path.dirname(in_dir), os.path.basename(in_dir)+"_audio")
    out_dir = os.path.join(os.path.dirname(in_dir), os.path.basename(in_dir)+"_preprocessed")
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    print("Gather audio...")
    video_files = glob(f'{in_dir}/**/*.mp4', recursive=True)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    filelist = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    speaker_info, speaker_done = dict(), set()
    total_duration = 0

    print("Create dataset...")
    for video_path in tqdm(video_files):
        # Load annotation file
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = video_path.replace('mp4', 'json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
        except UnicodeDecodeError:
            continue

        # Load video clip
        audio_path = video_path.replace(in_dir, audio_dir, 1).replace('mp4', 'wav')
        orig_sr = librosa.get_samplerate(audio_path)
        y, sr = librosa.load(audio_path, sr=orig_sr)
        duration = librosa.get_duration(y, sr=sr)
        new_sr = sampling_rate
        new_y = librosa.resample(y, sr, new_sr)

        # Metadata
        n_frames = float(annotation['nr_frame'])
        fps = n_frames / duration
        for spk_id, spk_info in annotation['actor'].items():
            if spk_id not in speaker_done:
                speaker_info[spk_id] = spk_info
                speaker_done.add(spk_id)

        turn_id = 0
        done = set()
        for frame, frame_data in get_sorted_items(annotation['data'].items()):
            for sub_id, info_data in frame_data.items():
                if 'text' not in info_data.keys():
                    continue

                # Extract data
                text_data = info_data['text']
                emotion_data = info_data['emotion']
                speaker_id = info_data['person_id']
                start_frame = text_data['script_start']
                end_frame = text_data['script_end']
                intent = text_data['intent']
                strategy = text_data['strategy']
                
                et_e, et_a, et_v = get_emotion(emotion_data['text'])
                es_e, es_a, es_v = get_emotion(emotion_data['sound'])
                ei_e, ei_a, ei_v = get_emotion(emotion_data['image'])
                em_e, em_a, em_v = get_emotion(emotion_data['multimodal'])

                script = refine_text(text_data['script'])

                start_idx = int(float(start_frame) / fps * new_sr)
                end_idx = int(float(end_frame) / fps * new_sr)

                # Write wav
                y_part = new_y[start_idx:end_idx]
                speaker_id = pad_spk_id(speaker_id)
                file_name = file_name.replace('clip_', 'c')
                framename = f'{start_frame}-{end_frame}'
                basename = f'{turn_id}_{speaker_id}_{file_name}_{framename}'
                wav_path = os.path.join(os.path.dirname(audio_path).replace(audio_dir, out_dir),
                                        f'{basename}.wav')
                if framename not in done:
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                    soundfile.write(wav_path, y_part, new_sr)
                    write_text(wav_path.replace('.wav', '.txt'), script)

                    # Write filelist
                    filelist.write(f'{basename}|{script}|{speaker_id}|{intent}|{strategy}|{et_e}|{et_a}|{et_v}|{es_e}|{es_a}|{es_v}|{ei_e}|{ei_a}|{ei_v}|{em_e}|{em_a}|{em_v}\n')
                    total_duration += (end_idx - start_idx) / float(new_sr)

                    done.add(f'{framename}')
                    turn_id += 1

    filelist.close()

    # Save Speaker Info
    with open(f'{out_dir}/speaker_info.txt', 'w', encoding='utf-8') as f:
        for spk_id, spk_info in get_sorted_items(speaker_info.items()):
            gender = 'F' if spk_info['gender'] == 'female' else 'M'
            age = spk_info['age']
            spk_id = pad_spk_id(speaker_id)
            f.write(f'{spk_id}|{gender}|{age}\n')

    print(f'End parsing, total duration: {total_duration}')


def refine_text(text):
    # Fix invalid characters in text
    text = text.replace('…', ',')
    text = text.replace('\t', '')
    text = text.replace('-', ',')
    text = text.replace('–', ',')
    text = ' '.join(text.split())
    return text


def extract_audio(preprocess_config):
    in_dir = preprocess_config["path"]["corpus_path"]
    out_dir = os.path.join(os.path.dirname(in_dir), os.path.basename(in_dir)+"_tmp")
    video_files = glob(f'{in_dir}/**/*.mp4', recursive=True)

    print("Extract audio...")
    for video_path in tqdm(video_files):
        audio_path = video_path.replace(in_dir, out_dir, 1).replace('mp4', 'wav')
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, verbose=False)
        clip.close()


def extract_nonkr(preprocess_config):
    in_dir = preprocess_config["path"]["raw_path"]
    filelist = open(f'{in_dir}/nonkr.txt', 'w', encoding='utf-8')

    count = 0
    nonkr = set()
    print("Extract non korean charactors...")
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_count = len(lines)
        for line in tqdm(lines):
            wav = line.split('|')[0]
            text = line.split('|')[1]
            reg = re.compile("""[^ ㄱ-ㅣ가-힣~!.,?:{}`"'＂“‘’”’()\[\]]+""")
            impurities = reg.findall(text)
            if len(impurities) == 0:
                count+=1
                continue
            norm = _clean_text(text, preprocess_config["preprocessing"]["text"]["text_cleaners"])
            impurities_str = ','.join(impurities)
            filelist.write(f'{norm}|{text}|{impurities_str}|{wav}\n')
            for imp in impurities:
                nonkr.add(imp)
    filelist.close()
    print('Total {} non korean charactors from {} lines'.format(len(nonkr), total_count-count))
    print(sorted(list(nonkr)))


def extract_lexicon(preprocess_config):
    """
    Extract lexicon and build grapheme-phoneme dictionary for MFA training
    See https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch
    """
    in_dir = preprocess_config["path"]["raw_path"]
    lexicon_path = preprocess_config["path"]["lexicon_path"]
    filelist = open(lexicon_path, 'a+', encoding='utf-8')

    # Load Lexicon Dictionary
    done = set()
    if os.path.isfile(lexicon_path):
        filelist.seek(0)
        for line in filelist.readlines():
            grapheme = line.split("\t")[0]
            done.add(grapheme)

    print("Extract lexicon...")
    for lab in tqdm(glob(f'{in_dir}/**/*.lab', recursive=True)):
        with open(lab, 'r', encoding='utf-8') as f:
            text = f.readline().strip("\n")
        assert text == normalize_nonchar(text), "No special token should be left."

        for grapheme in text.split(" "):
            if not grapheme in done:
                phoneme = " ".join(tokenize(grapheme, norm=False))
                filelist.write("{}\t{}\n".format(grapheme, phoneme))
                done.add(grapheme)
    filelist.close()


def apply_fixed_text(preprocess_config):
    in_dir = preprocess_config["path"]["corpus_path"]
    sub_dir = preprocess_config["path"]["sub_dir_name"]
    out_dir = preprocess_config["path"]["raw_path"]
    fixed_text_path = preprocess_config["path"]["fixed_text_path"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    fixed_text_dict = dict()
    print("Fixing transcripts...")
    with open(fixed_text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            wav, fixed_text = line.split('|')[0], line.split('|')[1]
            clip_name = wav.split('_')[2].replace('c', 'clip_')
            fixed_text_dict[wav] = fixed_text.replace('\n', '')

            text = _clean_text(fixed_text, cleaners)
            with open(
                os.path.join(out_dir, sub_dir, clip_name, "{}.lab".format(wav)),
                "w",
            ) as f1:
                f1.write(text)

    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as filelist:
        for line in tqdm(filelist.readlines()):
            wav = line.split('|')[0]
            if wav in fixed_text_dict:
                filelist_fixed.write("|".join([line.split("|")[0]] + [fixed_text_dict[wav]] + line.split("|")[2:]))
            else:
                filelist_fixed.write(line)
    filelist_fixed.close()

    extract_lexicon(preprocess_config)