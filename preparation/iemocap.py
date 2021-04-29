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
from text.korean import normalize_nonchar
from g2p_en import G2p


def extract_nonen(preprocess_config):
    in_dir = preprocess_config["path"]["raw_path"]
    filelist = open(f'{in_dir}/nonen.txt', 'w', encoding='utf-8')

    count = 0
    nonen = set()
    print("Extract non english charactors...")
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_count = len(lines)
        for line in tqdm(lines):
            wav = line.split('|')[0]
            text = line.split('|')[1]

            reg = re.compile("""[^ a-zA-Z~!.,?:`"'＂“‘’”’]+""")
            impurities = reg.findall(text)
            if len(impurities) == 0:
                count+=1
                continue
            norm = _clean_text(text, preprocess_config["preprocessing"]["text"]["text_cleaners"])
            impurities_str = ','.join(impurities)
            filelist.write(f'{norm}|{text}|{impurities_str}|{wav}\n')
            for imp in impurities:
                nonen.add(imp)
    filelist.close()
    print('Total {} non english charactors from {} lines'.format(len(nonen), total_count-count))
    print(sorted(list(nonen)))


def extract_lexicon(preprocess_config):
    """
    Extract lexicon and build grapheme-phoneme dictionary for MFA training
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
    g2p = G2p()
    for lab in tqdm(glob(f'{in_dir}/**/*.lab', recursive=True)):
        with open(lab, 'r', encoding='utf-8') as f:
            text = f.readline().strip("\n")
        text = normalize_nonchar(text)

        for grapheme in text.split(" "):
            if not grapheme in done:
                phoneme = " ".join(g2p(grapheme))
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
            session = '_'.join(wav.split('_')[1:])
            fixed_text_dict[wav] = fixed_text.replace('\n', '')

            text = _clean_text(fixed_text, cleaners)
            with open(
                os.path.join(out_dir, sub_dir, session, "{}.lab".format(wav)),
                "w",
            ) as f1:
                f1.write(text)

    filelist_fixed = open(f'{out_dir}/filelist_fixed.txt', 'w', encoding='utf-8')
    with open(f'{out_dir}/filelist.txt', 'r', encoding='utf-8') as filelist:
        for line in tqdm(filelist.readlines()):
            wav = line.split('|')[0]
            if wav in fixed_text_dict:
                filelist_fixed.write("|".join([line.split("|")[0]] + [fixed_text_dict[wav]] + line.split("|")[2:]))
            else:
                filelist_fixed.write(line)
    filelist_fixed.close()

    os.remove(f'{out_dir}/filelist.txt')
    os.rename(f'{out_dir}/filelist_fixed.txt', f'{out_dir}/filelist.txt')

    extract_lexicon(preprocess_config)