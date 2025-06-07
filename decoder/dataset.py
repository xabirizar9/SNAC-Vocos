from dataclasses import dataclass
from datasets import load_dataset

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import soundfile

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        # Load huggingface dataset with proper caching for distributed training
        try:
            self.dataset = load_dataset("xabirizar9/tara-high-pitch", split="train", 
                                      cache_dir=None, keep_in_memory=False)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to a minimal dataset for testing
            self.dataset = []

        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            # Handle empty dataset case
            if len(self.dataset) == 0:
                return torch.zeros(self.num_samples)
            
            audio_data = self.dataset[index]['audio']
            y = torch.from_numpy(audio_data['array']).unsqueeze(0).to(torch.float32) # (1, T)
            sr = audio_data['sampling_rate']

            # Handle empty or invalid audio
            if y.size(-1) == 0:
                return torch.zeros(self.num_samples)

            if y.ndim > 2:
                y = y.mean(dim=-1, keepdim=False)
            
            # Apply gain normalization with fallback if sox fails
            gain_db = np.random.uniform(-1, -6) if self.train else -3
            try:
                y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain_db:.2f}"]])
            except Exception:
                # Fallback: manual gain normalization
                gain_linear = 10 ** (gain_db / 20)
                y = y * gain_linear
                # Normalize to prevent clipping
                y = y / torch.max(torch.abs(y)).clamp(min=1e-8)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            if y.size(-1) < self.num_samples:
                pad_length = self.num_samples - y.size(-1)
                if y.size(-1) > 0:
                    padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                    y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
                else:
                    y = torch.zeros(1, self.num_samples)
            elif self.train:
                start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                y = y[:, start : start + self.num_samples]
            else:
                # During validation, take always the first segment for determinism
                y = y[:, : self.num_samples]

            return y[0]
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            # Return zeros as fallback
            return torch.zeros(self.num_samples)
