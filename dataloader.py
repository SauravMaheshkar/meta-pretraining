import ast
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wfdb

import wandb
from utils import set_seed

RawData = Union[List, np.ndarray]

__all__ = ["ECGDataSetWrapper"]

# ======================== ECGSimCLR ======================== #


class ECGSimCLR(torch.utils.data.Dataset):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, transform: Any, simclr: bool = True
    ) -> None:
        super(ECGSimCLR, self).__init__()

        # Padding
        if x.shape[1] != 1024 and x.shape[1] == 1000:
            x = np.pad(x, [[0, 0], [0, 24], [0, 0]])

        # Typecast as Float32
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

        self.transform = transform
        self.simclr = simclr

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple:
        x = self.x[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.simclr:
            return x
        else:
            sample = (x, y)
            return sample


# ======================== SimCLRDataTransform ======================== #


class SimCLRDataTransform(object):
    def __init__(self, transform: Sequence[Callable]) -> None:
        super(SimCLRDataTransform, self).__init__()
        self.transform = transform

    def __call__(self, sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi: np.ndarray = sample.copy()
        for t in self.transform:
            xi = t(xi)

        xj: np.ndarray = sample.copy()
        for t in self.transform:
            xj = t(xj)

        return xi.astype(np.float32), xj.astype(np.float32)


# ======================== ECGDataSetWrapper ======================== #


class ECGDataSetWrapper(object):
    def __init__(self, batch_size: int, num_workers: int = 0) -> None:

        super(ECGDataSetWrapper, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_simclr_pipeline_transform(self) -> List[Callable]:
        def rand_crop_ecg(ecg: np.ndarray) -> np.ndarray:
            """Randomly Crops the ECG"""
            cropped_ecg: np.ndarray = ecg.copy()
            for j in range(ecg.shape[1]):
                crop_len = np.random.randint(len(ecg)) // 2
                crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
                cropped_ecg[crop_start : crop_start + crop_len, j] = 0
            return cropped_ecg

        def rand_add_noise(ecg: np.ndarray) -> np.ndarray:
            """Adds Random Noise to the ECG"""
            noise_frac: float = np.random.rand() * 0.1
            return ecg + noise_frac * ecg.std(axis=0) * np.random.randn(*ecg.shape)

        data_transforms = [rand_crop_ecg, rand_add_noise]
        return data_transforms

    def get_data_loaders(self, args) -> Sequence[Any]:

        wandb.run.use_artifact("train_labels:v0", type="train_data")
        wandb.run.use_artifact("test_labels:v0", type="train_data")

        def load_raw_data(df, sampling_rate: int, path: str) -> RawData:
            """Returns the signal descriptors of the raw waveform data"""
            if sampling_rate == 100:
                data: RawData = [wfdb.rdsamp(path + f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
            data = np.array([signal for signal, _ in data])
            return data

        def aggregate_diagnostic(y_dic: Dict) -> np.ndarray:
            tmp: np.ndarray = np.zeros(5)
            idxd: Dict = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}
            for key in y_dic.keys():
                if key in agg_df.index:
                    cls = agg_df.loc[key].diagnostic_class
                    tmp[idxd[cls]] = 1
            return tmp

        path: str = "physionet.org/files/ptb-xl/1.0.1/"
        sampling_rate: int = 100

        # load and convert annotation data
        Y: pd.DataFrame = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X: RawData = load_raw_data(Y, sampling_rate, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df: pd.DataFrame = pd.read_csv(path + "scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        # Apply diagnostic superclass
        Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

        # Split data into train and test
        test_fold: int = 10

        # Training Data
        X_train: np.ndarray = X[np.where(Y.strat_fold != test_fold)]
        y_train: pd.Series = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        y_train = np.stack(y_train, axis=0)

        # Testing Data
        X_test: np.ndarray = X[np.where(Y.strat_fold == test_fold)]
        y_test: pd.Series = Y[Y.strat_fold == test_fold].diagnostic_superclass
        y_test = np.stack(y_test, axis=0)

        # Data Augmentation
        data_augment = self.get_simclr_pipeline_transform()
        data_augment = SimCLRDataTransform(data_augment)  # type:ignore

        FT_TASKS: int = 5

        # Normalization
        meansig = np.mean(X_train.reshape(-1))
        stdsig = np.std(X_train.reshape(-1))
        X_train = (X_train - meansig) / stdsig
        X_test = (X_test - meansig) / stdsig

        # PreTraining Dataset
        pretrain_dataset = ECGSimCLR(X_train, y_train, data_augment)

        # Set Random Seed
        set_seed(args.seed)

        rng = np.random.RandomState(args.seed)
        idxs = np.arange(len(y_train))
        rng.shuffle(idxs)

        # Apply Sampling
        if args.ex >= 50:
            train_samp = int(0.8 * args.ex)
            val_samp = args.ex - train_samp
        else:
            if args.ex == 25:
                train_samp = 15
                val_samp = 10
            elif args.ex == 10:
                train_samp = val_samp = 5

        # Train and Validation Indexes after Sampling
        train_idxs = idxs[:train_samp]
        val_idxs = idxs[train_samp : train_samp + val_samp]

        # ======================== Train, Validation and Test Dataset ======================== #

        ft_train = ECGSimCLR(
            X_train[train_idxs], y_train[train_idxs], transform=None, simclr=False
        )
        ft_val = ECGSimCLR(
            X_train[val_idxs], y_train[val_idxs], transform=None, simclr=False
        )
        ft_test = ECGSimCLR(X_test, y_test, transform=None, simclr=False)

        # ======================== PreTrain, Train, Validation and Test Dataloader ======================== #

        pretrain_loader = torch.utils.data.DataLoader(
            dataset=pretrain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        ft_train_loader = torch.utils.data.DataLoader(
            dataset=ft_train, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        ft_val_loader = torch.utils.data.DataLoader(
            dataset=ft_val, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        ft_test_loader = torch.utils.data.DataLoader(
            dataset=ft_test, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        return (
            pretrain_loader,
            ft_train_loader,
            ft_val_loader,
            ft_test_loader,
            None,
            FT_TASKS,
        )
