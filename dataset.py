import json
import math
import os
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        text_emb_path = os.path.join(
            self.preprocessed_path,
            "text_emb",
            "{}-text_emb-{}.npy".format(speaker, basename),
        )
        text_emb = np.load(text_emb_path)
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        # History
        dialog = basename.split("_")[2].strip("c")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text_emb = list()
        history_speaker = list()
        history_basenames = sorted([tg_path.replace(".wav", "")\
            for tg_path in os.listdir(os.path.join(self.raw_path, "clips", f"clip_{dialog}"))\
            if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
        history_basenames = history_basenames[:turn][-history_len:]

        for i, h_basename in enumerate(history_basenames):
            h_idx = int(self.basename_to_id[h_basename])
            h_speaker = self.speaker[h_idx]
            h_speaker_id = self.speaker_map[h_speaker]
            h_text_emb_path = os.path.join(
                self.preprocessed_path,
                "text_emb",
                "{}-text_emb-{}.npy".format(h_speaker, h_basename),
            )
            h_text_emb = np.load(h_text_emb_path)
            
            history_text_emb.append(h_text_emb)
            history_speaker.append(h_speaker_id)

            # Padding
            if i == history_len-1 and history_len < self.max_history_len:
                self.pad_history(
                    self.max_history_len-history_len,
                    history_text_emb,
                    history_speaker,
                )
        if turn == 0:
            self.pad_history(
                self.max_history_len,
                history_text_emb,
                history_speaker,
            )

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "text_emb": text_emb,
            "history_len": history_len,
            "history_text_emb": history_text_emb,
            "history_speaker": history_speaker,
        }

        return sample

    def pad_history(self, 
            pad_size,
            history_text_emb,
            history_speaker,
        ):
        # meaningless zero padding, should be cut out by mask of history_len
        for _ in range(pad_size):
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32))
            history_speaker.append(0)

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in tqdm(f.readlines()):
                line_split = line.strip("\n").split("|")
                n, s, t, r = line_split[:4]
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        text_embs = [data[idx]["text_emb"] for idx in idxs]
        history_lens = [data[idx]["history_len"] for idx in idxs]
        history_text_embs = [data[idx]["history_text_emb"] for idx in idxs]
        history_speakers = [data[idx]["history_speaker"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        history_lens = np.array(history_lens)

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        text_embs = np.array(text_embs)
        history_text_embs = np.array(history_text_embs)
        history_speakers = np.array(history_speakers)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            text_embs,
            history_lens,
            history_text_embs,
            history_speakers,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.raw_path = preprocess_config["path"]["raw_path"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        text_emb_path = os.path.join(
            self.preprocessed_path,
            "text_emb",
            "{}-text_emb-{}.npy".format(speaker, basename),
        )
        text_emb = np.load(text_emb_path)

        # History
        dialog = basename.split("_")[2].strip("c")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text_emb = list()
        history_speaker = list()
        history_basenames = sorted([tg_path.replace(".wav", "")\
            for tg_path in os.listdir(os.path.join(self.raw_path, "clips", f"clip_{dialog}"))\
            if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
        history_basenames = history_basenames[:turn][-history_len:]

        for i, h_basename in enumerate(history_basenames):
            h_idx = int(self.basename_to_id[h_basename])
            h_speaker = self.speaker[h_idx]
            h_speaker_id = self.speaker_map[h_speaker]
            h_text_emb_path = os.path.join(
                self.preprocessed_path,
                "text_emb",
                "{}-text_emb-{}.npy".format(h_speaker, h_basename),
            )
            h_text_emb = np.load(h_text_emb_path)
            
            history_text_emb.append(h_text_emb)
            history_speaker.append(h_speaker_id)

            # Padding
            if i == history_len-1 and history_len < self.max_history_len:
                self.pad_history(
                    self.max_history_len-history_len,
                    history_text_emb,
                    history_speaker,
                )
        if turn == 0:
            self.pad_history(
                self.max_history_len,
                history_text_emb,
                history_speaker,
            )

        return (
            basename, 
            speaker_id, 
            phone, 
            raw_text,
            text_emb,
            history_len,
            history_text_emb,
            history_speaker,
        )

    def pad_history(self, 
            pad_size,
            history_text_emb,
            history_speaker,
        ):
        # meaningless zero padding, should be cut out by mask of history_len
        for _ in range(pad_size):
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32))
            history_speaker.append(0)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            aux_data = []
            for line in tqdm(f.readlines()):
                line_split = line.strip("\n").split("|")
                n, s, t, r = line_split[:4]
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        text_embs = np.array([d[4] for d in data])
        history_lens = [d[5] for d in data] # ["history_len"]
        history_lens = np.array(history_lens)        
        history_text_embs = [d[6] for d in data] # ["history_text_emb"]
        history_text_embs = np.array(history_text_embs)
        history_speakers = [d[7] for d in data] # ["history_speaker"]
        history_speakers = np.array(history_speakers)

        return (
            ids, 
            raw_texts, 
            speakers, 
            texts, 
            text_lens, 
            max(text_lens),
            text_embs, 
            history_lens, 
            history_text_embs, 
            history_speakers,
        )


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("./config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train_dialog.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val_dialog.txt", preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )