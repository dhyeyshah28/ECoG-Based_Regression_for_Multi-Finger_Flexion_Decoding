import numpy as np
import scipy.signal
import scipy.ndimage
import pywt
import pathlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
import os
import scipy.io
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


# Hyperparameters
L_FREQ, H_FREQ = 5, 200  # Frequency bounds
WAVELET_NUM = 40  # Number of wavelets
DOWNSAMPLE_FS = 100  # Target sampling rate
time_delay_secs = 0.037  # Time delay
SAMPLE_LEN = 256  # Window size
finger_num = 5  # Number of fingers
ORIGINAL_FS = 1000  # Assumed original sampling rate
BATCH_SIZE = 64  # Adjusted for RTX 4070
CHANNELS_NUM = [62, 48, 64]  # Channels per patient
MAX_EPOCHS = 10  # Training epochs



# Paths
SAVE_PATH = f"{pathlib.Path().resolve()}/data_again/"
CHECKPOINT_DIR = f"{pathlib.Path().resolve()}/checkpoints_again/"
RES_NPY_DIR = f"{pathlib.Path().resolve()}/results_again/"
pathlib.Path(SAVE_PATH + "train").mkdir(parents=True, exist_ok=True)
pathlib.Path(SAVE_PATH + "val").mkdir(parents=True, exist_ok=True)
pathlib.Path(SAVE_PATH + "test").mkdir(parents=True, exist_ok=True)
pathlib.Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
pathlib.Path(RES_NPY_DIR).mkdir(exist_ok=True)


def preprocess_ecog(ecog_data, original_fs=ORIGINAL_FS, target_fs=DOWNSAMPLE_FS):
    """Preprocess ECoG: bandpass filter, wavelet transform, downsample."""
    print("  Applying bandpass filter (40-300 Hz)...")
    sos = scipy.signal.butter(4, [L_FREQ, H_FREQ], btype='band', fs=original_fs, output='sos')
    ecog_filtered = scipy.signal.sosfiltfilt(sos, ecog_data, axis=0)

    print("  Performing continuous wavelet transform...")
    scales = pywt.scale2frequency('cmor1.5-1.0', np.arange(1, WAVELET_NUM + 1)) * original_fs
    scales = scales[(scales >= L_FREQ) & (scales <= H_FREQ)]
    ecog_wavelet = np.zeros((ecog_data.shape[1], WAVELET_NUM, ecog_data.shape[0]))
    for ch in range(ecog_data.shape[1]):
        coef, _ = pywt.cwt(ecog_filtered[:, ch], scales, 'cmor1.5-1.0', sampling_period=1 / original_fs)
        ecog_wavelet[ch, :coef.shape[0], :] = np.abs(coef)
    ecog_wavelet = ecog_wavelet[:, :WAVELET_NUM, :]

    print("  Downsampling to 100 Hz...")
    downsample_factor = int(original_fs // target_fs)
    ecog_downsampled = scipy.signal.decimate(ecog_wavelet, downsample_factor, axis=-1)
    return ecog_downsampled.astype(np.float32)


def preprocess_fingerflex(fingerflex_data, original_fs=ORIGINAL_FS, target_fs=DOWNSAMPLE_FS):
    """Downsample fingerflex data0."""
    print("  Downsampling fingerflex data0...")
    downsample_factor = int(original_fs // target_fs)
    fingerflex_downsampled = scipy.signal.decimate(fingerflex_data, downsample_factor, axis=0)
    return fingerflex_downsampled.astype(np.float32)


def save_preprocessed_data(data):
    """Preprocess and save data0 for each patient."""
    print("Preprocessing data0 for all patients...")
    for p in range(3):
        print(f"Processing Patient {p + 1}...")
        ecog = data['train_ecog'][p, 0]
        dg = data['train_dg'][p, 0]

        ecog_proc = preprocess_ecog(ecog)
        dg_proc = preprocess_fingerflex(dg)

        total_time = ecog_proc.shape[-1]
        train_idx = int(0.85 * total_time)
        val_idx = total_time

        ecog_train = ecog_proc[..., :train_idx]
        ecog_val = ecog_proc[..., train_idx:val_idx]
        ecog_test = ecog_proc[..., val_idx:total_time]
        dg_train = dg_proc[:train_idx, :]
        dg_val = dg_proc[train_idx:val_idx, :]
        dg_test = dg_proc[val_idx:, :]

        np.save(f"{SAVE_PATH}train/ecog_data_p{p + 1}.npy", ecog_train)
        np.save(f"{SAVE_PATH}train/fingerflex_data_p{p + 1}.npy", dg_train)
        np.save(f"{SAVE_PATH}val/ecog_data_p{p + 1}.npy", ecog_val)
        np.save(f"{SAVE_PATH}val/fingerflex_data_p{p + 1}.npy", dg_val)
        np.save(f"{SAVE_PATH}test/ecog_data_p{p + 1}.npy", ecog_test)
        np.save(f"{SAVE_PATH}test/fingerflex_data_p{p + 1}.npy", dg_test)
    print("Preprocessing complete.")


class EcogFingerflexDataset(Dataset):
    def __init__(self, path_to_ecog_data: str, path_to_fingerflex_data: str, sample_len: int, train=False):
        self.ecog_data = np.load(path_to_ecog_data).astype(np.float32)
        self.fingerflex_data = np.load(path_to_fingerflex_data).astype(np.float32)
        self.duration = self.ecog_data.shape[-1]
        self.sample_len = sample_len
        self.stride = 1
        self.ds_len = (self.duration - self.sample_len) // self.stride
        self.train = train

    def __len__(self):
        return self.ds_len

    def __getitem__(self, index):
        sample_start = index * self.stride
        sample_end = sample_start + self.sample_len
        ecog_sample = self.ecog_data[..., sample_start:sample_end]
        fingerflex_sample = self.fingerflex_data[sample_start:sample_end, :].transpose(1, 0)
        return torch.from_numpy(ecog_sample).float(), torch.from_numpy(fingerflex_sample).float()


class EcogFingerflexDatamodule(pl.LightningDataModule):
    def __init__(self, sample_len: int, data_dir=SAVE_PATH, batch_size=BATCH_SIZE, patient_id=1):
        super().__init__()
        self.data_dir = data_dir
        self.sample_len = sample_len
        self.batch_size = batch_size
        self.add_name = f"_p{patient_id}"
        self.test = None  # Initialize test dataset as None

    def setup(self, stage=None):
        print(f"Setting up datamodule for Patient {self.add_name[2:]}...")

        # Always setup train and val datasets
        self.train = EcogFingerflexDataset(
            f"{self.data_dir}train/ecog_data{self.add_name}.npy",
            f"{self.data_dir}train/fingerflex_data{self.add_name}.npy",
            self.sample_len, train=True
        )
        self.val = EcogFingerflexDataset(
            f"{self.data_dir}val/ecog_data{self.add_name}.npy",
            f"{self.data_dir}val/fingerflex_data{self.add_name}.npy",
            self.sample_len
        )

        # Only setup test dataset when needed
        if stage == "test" or stage is None:
            self.test = EcogFingerflexDataset(
                f"{self.data_dir}test/ecog_data{self.add_name}.npy",
                f"{self.data_dir}test/fingerflex_data{self.add_name}.npy",
                self.sample_len
            )
        print("Datamodule setup complete.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        if self.test is None:
            self.setup(stage="test")  # Ensure test dataset is initialized
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, p_conv_drop=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, padding='same')
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)
        self.stride = stride

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.drop(x)
        x = self.downsample(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, scale, **args):
        super().__init__()
        self.conv_block = ConvBlock(**args)
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class AutoEncoder1D(nn.Module):
    def __init__(self, n_electrodes=64, n_freqs=40, n_channels_out=5,
                 channels=[32, 32, 64, 64, 128, 128], kernel_sizes=[7, 7, 5, 5, 5],
                 strides=[2, 2, 2, 2, 2], dilation=[1, 1, 1, 1, 1]):
        super().__init__()
        self.n_inp_features = n_freqs * n_electrodes
        self.model_depth = len(channels) - 1
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
        self.downsample_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], kernel_sizes[i], strides[i], dilation[i])
            for i in range(self.model_depth)
        ])
        channels = channels[:-1] + channels[-1:]
        self.upsample_blocks = nn.ModuleList([
            UpConvBlock(scale=strides[i],
                        in_channels=channels[i + 1] if i == self.model_depth - 1 else channels[i + 1] * 2,
                        out_channels=channels[i], kernel_size=kernel_sizes[i])
            for i in range(self.model_depth - 1, -1, -1)
        ])
        self.conv1x1_one = nn.Conv1d(channels[0] * 2, n_channels_out, kernel_size=1, padding='same')

    def forward(self, x):
        batch, elec, n_freq, time = x.shape
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        skip_connection = []
        for i in range(self.model_depth):
            skip_connection.append(x)
            x = self.downsample_blocks[i](x)
        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat((x, skip_connection[-1 - i]), dim=1)
        x = self.conv1x1_one(x)
        return x


class BaseEcogFingerflexModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 8.42e-5
        self.all_metrics = {
            'train': {f'finger_{i}': [] for i in range(finger_num)},
            'val': {f'finger_{i}': [] for i in range(finger_num)},
            'test': {f'finger_{i}': [] for i in range(finger_num)}
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        corr = correlation_metric(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_corr", corr, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        corr = correlation_metric(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_corr", corr, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_corr': corr, 'preds': y_hat, 'targets': y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        corr = correlation_metric(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_corr", corr, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_corr': corr, 'preds': y_hat, 'targets': y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)


class ComprehensiveEvaluationCallback(Callback):
    def __init__(self, patient_id):
        super().__init__()
        self.patient_id = patient_id
        self.results = {
            'train': {'predictions': [], 'targets': []},
            'val': {'predictions': [], 'targets': []},
            'test': {'predictions': [], 'targets': []}
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = pl_module.model(x)
        self.results['train']['predictions'].append(y_hat.cpu().numpy())
        self.results['train']['targets'].append(y.cpu().numpy())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = pl_module.model(x)
        self.results['val']['predictions'].append(y_hat.cpu().numpy())
        self.results['val']['targets'].append(y.cpu().numpy())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = pl_module.model(x)
        self.results['test']['predictions'].append(y_hat.cpu().numpy())
        self.results['test']['targets'].append(y.cpu().numpy())

    def on_train_end(self, trainer, pl_module):
        self._process_and_save_results('train', pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._process_and_save_results('val', pl_module)

    def on_test_end(self, trainer, pl_module):
        self._process_and_save_results('test', pl_module)

    def _process_and_save_results(self, dataset_type, pl_module):
        preds = np.concatenate(self.results[dataset_type]['predictions'], axis=0)
        targets = np.concatenate(self.results[dataset_type]['targets'], axis=0)

        correlations = []
        for finger in range(finger_num):
            pred_finger = preds[:, finger, :].flatten()
            target_finger = targets[:, finger, :].flatten()
            corr = np.corrcoef(pred_finger, target_finger)[0, 1]
            correlations.append(corr)
            pl_module.all_metrics[dataset_type][f'finger_{finger}'].append(corr)

        np.save(f"{RES_NPY_DIR}{dataset_type}_predictions_p{self.patient_id}.npy", preds)
        np.save(f"{RES_NPY_DIR}{dataset_type}_targets_p{self.patient_id}.npy", targets)

        self._plot_results(dataset_type, preds, targets, correlations)

    def _plot_results(self, dataset_type, preds, targets, correlations):
        plt.figure(figsize=(15, 10))
        for finger in range(finger_num):
            plt.subplot(3, 2, finger + 1)
            pred_finger = preds[:, finger, :].flatten()
            target_finger = targets[:, finger, :].flatten()

            plt.plot(pred_finger[:1000], label='Predicted', alpha=0.7)
            plt.plot(target_finger[:1000], label='Actual', alpha=0.7)
            plt.title(f"Finger {finger} - Corr: {correlations[finger]:.3f}")
            plt.legend()

        plt.suptitle(f"{dataset_type.capitalize()} Results - Patient {self.patient_id}")
        plt.tight_layout()
        plt.savefig(f"{RES_NPY_DIR}{dataset_type}_results_p{self.patient_id}.png")
        plt.close()


def correlation_metric(x, y):
    x = x.flatten()
    y = y.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr


def corr_metric(x, y):
    x, y = x.flatten(), y.flatten()
    return np.corrcoef(x, y)[0, 1]


def plot_final_comparison(all_patient_results):
    plt.figure(figsize=(15, 10))

    for dataset_idx, dataset_type in enumerate(['train', 'val', 'test']):
        for patient_id in range(1, 4):
            plt.subplot(3, 3, dataset_idx * 3 + patient_id)

            patient_key = f"patient_{patient_id}"
            metrics = all_patient_results[patient_key][dataset_type]

            finger_corrs = []
            for finger in range(finger_num):
                finger_key = f"finger_{finger}"
                if metrics[finger_key]:
                    finger_corrs.append(np.mean(metrics[finger_key]))
                else:
                    finger_corrs.append(0)

            plt.bar(range(finger_num), finger_corrs)
            plt.xticks(range(finger_num), [f'Finger {i}' for i in range(finger_num)])
            plt.ylim(0, 1)
            plt.title(f"Patient {patient_id} - {dataset_type.capitalize()}")
            plt.ylabel("Mean Correlation")

    plt.suptitle("Final Performance Comparison Across Patients")
    plt.tight_layout()
    plt.savefig(f"{RES_NPY_DIR}final_comparison.png")
    plt.close()


def main():
    print("Starting BCI training pipeline...")
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Please check GPU drivers and PyTorch installation.")
        return
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # Load and preprocess data0
    try:
        data = scipy.io.loadmat("C:\\Users\\dhyey\\OneDrive\\Desktop\\MSE_ROBO\\Sem_2\\Brain_Computer_Interfaces\\Assignments\\Final_Project\\raw_training_data.mat")
        if 'train_ecog' not in data or 'train_dg' not in data:
            raise KeyError("Dataset must contain 'train_ecog' and 'train_dg' keys")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Save preprocessed data0 if not already exists
    if not os.path.exists(f"{SAVE_PATH}train/ecog_data_p1.npy"):
        print("Preprocessing data0...")
        save_preprocessed_data(data)
    else:
        print("Preprocessed data0 already exists, skipping preprocessing...")

    all_patient_results = {}
    for patient_id in range(1, 4):
        print(f"\n=== Processing Patient {patient_id} ===")

        # Initialize model with patient-specific parameters
        hp_autoencoder = {
            "channels": [32, 32, 64, 64, 128, 128],
            "kernel_sizes": [7, 7, 5, 5, 5],
            "strides": [2, 2, 2, 2, 2],
            "dilation": [1, 1, 1, 1, 1],
            "n_electrodes": CHANNELS_NUM[patient_id - 1],
            "n_freqs": WAVELET_NUM,
            "n_channels_out": finger_num
        }
        model = AutoEncoder1D(**hp_autoencoder).to("cuda")
        lightning_wrapper = BaseEcogFingerflexModel(model)

        # Initialize data0 module
        dm = EcogFingerflexDatamodule(
            sample_len=SAMPLE_LEN,
            patient_id=patient_id
        )

        # Setup data0 - explicitly for training first
        dm.setup(stage="fit")

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor="val_corr",
            mode="max",
            dirpath=CHECKPOINT_DIR,
            filename=f"model_p{patient_id}-{{epoch:02d}}-{{val_corr:.4f}}"
        )

        early_stop_callback = EarlyStopping(
            monitor="val_corr",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="max"
        )

        eval_callback = ComprehensiveEvaluationCallback(patient_id)

        # Trainer configuration
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=MAX_EPOCHS,
            callbacks=[checkpoint_callback, early_stop_callback, eval_callback],
            logger=True,
            enable_progress_bar=True
        )

        # Training
        print(f"\nTraining Patient {patient_id} model...")
        trainer.fit(lightning_wrapper, dm)

        # Testing - ensure test dataset is properly setup
        print(f"\nTesting Patient {patient_id} model...")
        try:
            dm.setup(stage="test")  # Explicitly setup test data0
            test_loader = dm.test_dataloader()
            if len(test_loader.dataset) == 0:
                raise ValueError("Test dataset is empty")

            trainer.test(
                lightning_wrapper,
                dataloaders=test_loader,
                ckpt_path="best"  # Use best checkpoint from training
            )
        except Exception as e:
            print(f"Error during testing for Patient {patient_id}: {e}")
            continue

        # Store results0
        all_patient_results[f"patient_{patient_id}"] = lightning_wrapper.all_metrics

    # Generate final comparison plots
    print("\nGenerating final comparison plots...")
    plot_final_comparison(all_patient_results)
    print("\nTraining and evaluation complete!")



if __name__ == "__main__":
    main()








# import numpy as np
# import scipy.signal
# import scipy.ndimage
# import pywt
# import pathlib
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data0 import Dataset, DataLoader
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import Callback, ModelCheckpoint
# import os
#
# # Optimize for RTX 4070 Tensor Cores
# torch.set_float32_matmul_precision('high')
#
# # Hyperparameters
# L_FREQ, H_FREQ = 40, 300  # Frequency bounds
# WAVELET_NUM = 40          # Number of wavelets
# DOWNSAMPLE_FS = 100       # Target sampling rate
# time_delay_secs = 0.04     # Time delay
# SAMPLE_LEN = 256          # Window size
# finger_num = 5            # Number of fingers
# ORIGINAL_FS = 1000        # Assumed original sampling rate
# BATCH_SIZE = 64           # Adjusted for RTX 4070
# CHANNELS_NUM = [62, 48, 64]  # Channels per patient
# MAX_EPOCHS = 5           # Training epochs
#
# # Paths
# SAVE_PATH = f"{pathlib.Path().resolve()}/data0/"
# CHECKPOINT_DIR = f"{pathlib.Path().resolve()}/checkpoints0/"
# RES_NPY_DIR = f"{pathlib.Path().resolve()}/res_npy/"
# pathlib.Path(SAVE_PATH + "train").mkdir(parents=True, exist_ok=True)
# pathlib.Path(SAVE_PATH + "val").mkdir(parents=True, exist_ok=True)
# pathlib.Path(SAVE_PATH + "test").mkdir(parents=True, exist_ok=True)
# pathlib.Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
# pathlib.Path(RES_NPY_DIR).mkdir(exist_ok=True)
#
# # Script mode
# TYPE = "train"
#
# # Preprocessing functions
# def preprocess_ecog(ecog_data, original_fs=ORIGINAL_FS, target_fs=DOWNSAMPLE_FS):
#     """Preprocess ECoG: bandpass filter, wavelet transform, downsample."""
#     print("  Applying bandpass filter (40-300 Hz)...")
#     sos = scipy.signal.butter(4, [L_FREQ, H_FREQ], btype='band', fs=original_fs, output='sos')
#     ecog_filtered = scipy.signal.sosfiltfilt(sos, ecog_data, axis=0)
#
#     print("  Performing continuous wavelet transform...")
#     scales = pywt.scale2frequency('cmor1.5-1.0', np.arange(1, WAVELET_NUM+1)) * original_fs
#     scales = scales[(scales >= L_FREQ) & (scales <= H_FREQ)]
#     ecog_wavelet = np.zeros((ecog_data.shape[1], WAVELET_NUM, ecog_data.shape[0]))
#     for ch in range(ecog_data.shape[1]):
#         coef, _ = pywt.cwt(ecog_filtered[:, ch], scales, 'cmor1.5-1.0', sampling_period=1/original_fs)
#         ecog_wavelet[ch, :coef.shape[0], :] = np.abs(coef)
#     ecog_wavelet = ecog_wavelet[:, :WAVELET_NUM, :]
#
#     print("  Downsampling to 100 Hz...")
#     downsample_factor = int(original_fs // target_fs)
#     ecog_downsampled = scipy.signal.decimate(ecog_wavelet, downsample_factor, axis=-1)
#     return ecog_downsampled.astype(np.float32)
#
# def preprocess_fingerflex(fingerflex_data, original_fs=ORIGINAL_FS, target_fs=DOWNSAMPLE_FS):
#     """Downsample fingerflex data0."""
#     print("  Downsampling fingerflex data0...")
#     downsample_factor = int(original_fs // target_fs)
#     fingerflex_downsampled = scipy.signal.decimate(fingerflex_data, downsample_factor, axis=0)
#     return fingerflex_downsampled.astype(np.float32)
#
# def save_preprocessed_data(data0):
#     """Preprocess and save data0 for each patient: train (70%), val (15%), test (15%)."""
#     print("Preprocessing data0 for all patients...")
#     for p in range(3):
#         print(f"Processing Patient {p+1}...")
#         try:
#             ecog = data0['train_ecog'][p, 0]  # (300000, channels)
#             dg = data0['train_dg'][p, 0]      # (300000, 5)
#             print(f"  ECoG shape: {ecog.shape}, Fingerflex shape: {dg.shape}")
#         except KeyError as e:
#             print(f"Error: Missing key {e} in data0 dictionary")
#             raise
#
#         # Preprocess
#         ecog_proc = preprocess_ecog(ecog)
#         dg_proc = preprocess_fingerflex(dg)
#         print(f"  Preprocessed ECoG shape: {ecog_proc.shape}, Preprocessed Fingerflex shape: {dg_proc.shape}")
#
#         # Split: 70% train, 15% val, 15% test
#         total_time = ecog_proc.shape[-1]
#         train_idx = int(0.7 * total_time)
#         val_idx = int(0.85 * total_time)
#         ecog_train = ecog_proc[..., :train_idx]
#         ecog_val = ecog_proc[..., train_idx:val_idx]
#         ecog_test = ecog_proc[..., val_idx:total_time]
#         dg_train = dg_proc[:train_idx, :]
#         dg_val = dg_proc[train_idx:val_idx, :]
#         dg_test = dg_proc[val_idx:, :]
#         print(f"  Train: {ecog_train.shape[-1]} time points, Val: {ecog_val.shape[-1]} time points, Test: {ecog_test.shape[-1]} time points")
#
#         # Save
#         print(f"  Saving preprocessed data0 for Patient {p+1}...")
#         np.save(f"{SAVE_PATH}train/ecog_data_p{p+1}.npy", ecog_train)
#         np.save(f"{SAVE_PATH}train/fingerflex_data_p{p+1}.npy", dg_train)
#         np.save(f"{SAVE_PATH}val/ecog_data_p{p+1}.npy", ecog_val)
#         np.save(f"{SAVE_PATH}val/fingerflex_data_p{p+1}.npy", dg_val)
#         np.save(f"{SAVE_PATH}test/ecog_data_p{p+1}.npy", ecog_test)
#         np.save(f"{SAVE_PATH}test/fingerflex_data_p{p+1}.npy", dg_test)
#     print("Preprocessing complete.")
#
# # Dataset
# class EcogFingerflexDataset(Dataset):
#     def __init__(self, path_to_ecog_data: str, path_to_fingerflex_data: str, sample_len: int, train=False):
#         print(f"Loading dataset: {path_to_ecog_data}")
#         self.ecog_data = np.load(path_to_ecog_data).astype(np.float32)
#         self.fingerflex_data = np.load(path_to_fingerflex_data).astype(np.float32)
#         print(f"  ECoG data0 shape: {self.ecog_data.shape}, Fingerflex data0 shape: {self.fingerflex_data.shape}")
#         self.duration = self.ecog_data.shape[-1]
#         self.sample_len = sample_len
#         self.stride = 1
#         self.ds_len = (self.duration - self.sample_len) // self.stride
#         self.train = train
#         print(f"Dataset: Duration={self.duration}, Samples={self.ds_len}")
#
#     def __len__(self):
#         return self.ds_len
#
#     def __getitem__(self, index):
#         sample_start = index * self.stride
#         sample_end = sample_start + self.sample_len
#         ecog_sample = self.ecog_data[..., sample_start:sample_end]
#         fingerflex_sample = self.fingerflex_data[sample_start:sample_end, :].transpose(1, 0)  # Shape: (5, 256)
#         # Convert to PyTorch tensors
#         ecog_sample = torch.from_numpy(ecog_sample).float()
#         fingerflex_sample = torch.from_numpy(fingerflex_sample).float()
#         return ecog_sample, fingerflex_sample
#
# class EcogFingerflexDatamodule(pl.LightningDataModule):
#     def __init__(self, sample_len: int, data_dir=SAVE_PATH, batch_size=BATCH_SIZE, patient_id=1):
#         super().__init__()
#         self.data_dir = data_dir
#         self.sample_len = sample_len
#         self.batch_size = batch_size
#         self.add_name = f"_p{patient_id}"
#
#     def setup(self, stage=None):
#         print(f"Setting up datamodule for Patient {self.add_name[2:]}...")
#         if stage in (None, "fit"):
#             self.train = EcogFingerflexDataset(
#                 f"{self.data_dir}train/ecog_data{self.add_name}.npy",
#                 f"{self.data_dir}train/fingerflex_data{self.add_name}.npy",
#                 self.sample_len, train=True
#             )
#             self.val = EcogFingerflexDataset(
#                 f"{self.data_dir}val/ecog_data{self.add_name}.npy",
#                 f"{self.data_dir}val/fingerflex_data{self.add_name}.npy",
#                 self.sample_len
#             )
#         if stage in (None, "test"):
#             self.test = EcogFingerflexDataset(
#                 f"{self.data_dir}test/ecog_data{self.add_name}.npy",
#                 f"{self.data_dir}test/fingerflex_data{self.add_name}.npy",
#                 self.sample_len
#             )
#         print("Datamodule setup complete.")
#
#     def train_dataloader(self):
#         return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, shuffle=True)
#
#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)
#
#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)
#
# # Metrics
# def correlation_metric(x, y):
#     cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)
#     return torch.mean(cos_metric(x, y))
#
# def corr_metric(x, y):
#     x, y = x.flatten(), y.flatten()
#     return np.corrcoef(x, y)[0, 1]
#
# # Model
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, p_conv_drop=0.1):
#         super().__init__()
#         self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, padding='same')
#         self.norm = nn.LayerNorm(out_channels)
#         self.activation = nn.GELU()
#         self.drop = nn.Dropout(p=p_conv_drop)
#         self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)
#         self.stride = stride
#
#     def forward(self, x):
#         # print(f"ConvBlock input shape: {x.shape}")
#         x = self.conv1d(x)
#         # print(f"ConvBlock after conv1d shape: {x.shape}")
#         x = torch.transpose(x, -2, -1)
#         x = self.norm(x)
#         x = torch.transpose(x, -2, -1)
#         x = self.activation(x)
#         x = self.drop(x)
#         x = self.downsample(x)
#         return x
#
# class UpConvBlock(nn.Module):
#     def __init__(self, scale, **args):
#         super().__init__()
#         self.conv_block = ConvBlock(**args)
#         self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)
#
#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.upsample(x)
#         return x
#
# class AutoEncoder1D(nn.Module):
#     def __init__(self, n_electrodes=64, n_freqs=40, n_channels_out=5,
#                  channels=[32, 32, 64, 64, 128, 128], kernel_sizes=[7, 7, 5, 5, 5],
#                  strides=[2, 2, 2, 2, 2], dilation=[1, 1, 1, 1, 1]):
#         super().__init__()
#         self.n_inp_features = n_freqs * n_electrodes
#         self.model_depth = len(channels) - 1
#         self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
#         self.downsample_blocks = nn.ModuleList([
#             ConvBlock(channels[i], channels[i+1], kernel_sizes[i], strides[i], dilation[i])
#             for i in range(self.model_depth)
#         ])
#         channels = channels[:-1] + channels[-1:]
#         self.upsample_blocks = nn.ModuleList([
#             UpConvBlock(scale=strides[i], in_channels=channels[i+1] if i == self.model_depth-1 else channels[i+1]*2,
#                         out_channels=channels[i], kernel_size=kernel_sizes[i])
#             for i in range(self.model_depth-1, -1, -1)
#         ])
#         self.conv1x1_one = nn.Conv1d(channels[0]*2, n_channels_out, kernel_size=1, padding='same')
#
#     def forward(self, x):
#         # print(f"AutoEncoder1D input shape: {x.shape}")
#         batch, elec, n_freq, time = x.shape
#         x = x.reshape(batch, -1, time)
#         # print(f"AutoEncoder1D after reshape shape: {x.shape}")
#         x = self.spatial_reduce(x)
#         skip_connection = []
#         for i in range(self.model_depth):
#             skip_connection.append(x)
#             x = self.downsample_blocks[i](x)
#         for i in range(self.model_depth):
#             x = self.upsample_blocks[i](x)
#             x = torch.cat((x, skip_connection[-1 - i]), dim=1)
#         x = self.conv1x1_one(x)
#         return x
#
# class BaseEcogFingerflexModel(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.lr = 8.42e-5
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = F.mse_loss(y_hat, y)
#         corr = correlation_metric(y_hat, y)
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("cosine_dst_train", corr, on_step=False, on_epoch=True, prog_bar=True)
#         return 0.5 * loss + 0.5 * (1. - corr)
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = F.mse_loss(y_hat, y)
#         corr = correlation_metric(y_hat, y)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("cosine_dst_val", corr, on_step=False, on_epoch=True, prog_bar=True)
#         return y_hat
#
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = F.mse_loss(y_hat, y)
#         self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         return y_hat
#
#     def on_train_epoch_end(self):
#         metrics = self.trainer.logged_metrics
#         epoch = self.current_epoch
#         train_loss = metrics.get("train_loss", 0.0)
#         cosine_dst_train = metrics.get("cosine_dst_train", 0.0)
#         print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f}, Cosine Similarity (Train): {cosine_dst_train:.4f}")
#
#     def on_validation_epoch_end(self):
#         metrics = self.trainer.logged_metrics
#         epoch = self.current_epoch
#         val_loss = metrics.get("val_loss", 0.0)
#         cosine_dst_val = metrics.get("cosine_dst_val", 0.0)
#         corr_mean_val = metrics.get("corr_mean_val", 0.0)
#         print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Val Loss: {val_loss:.4f}, Cosine Similarity (Val): {cosine_dst_val:.4f}, Mean Correlation: {corr_mean_val:.4f}")
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
#
# # Callbacks
# class ValidationCallback(Callback):
#     def __init__(self, val_x, val_y, fg_num, patient_id):
#         super().__init__()
#         self.val_x = val_x  # (channels, wavelets, time), e.g., (62, 40, 4500)
#         self.val_y = val_y.T  # (time, fingers)
#         self.fg_num = fg_num
#         self.patient_id = patient_id
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         with torch.no_grad():
#             SIZE = 64
#             bound = self.val_x.shape[-1] // SIZE * SIZE
#             X_test = self.val_x[..., :bound]  # (channels, wavelets, time)
#             print(f"ValidationCallback X_test shape: {X_test.shape}")
#             y_test = self.val_y[:bound, :]
#             x_batch = torch.from_numpy(X_test).float().to(pl_module.device)
#             x_batch = x_batch.unsqueeze(0)  # (1, channels, wavelets, time)
#             print(f"ValidationCallback x_batch shape after unsqueeze: {x_batch.shape}")
#             y_hat = pl_module.model(x_batch)[0].cpu().numpy()
#             print(f"ValidationCallback y_hat shape: {y_hat.shape}")
#             y_prediction = scipy.ndimage.gaussian_filter1d(y_hat.T, sigma=6).T
#             y_test = y_test[::int(DOWNSAMPLE_FS/100), :]
#
#             h, w = self.fg_num // 2, self.fg_num - self.fg_num // 2
#             fig, ax = plt.subplots(h, w, figsize=(h*5, w*6), sharex=True, sharey=True)
#             corrs = []
#             metric_log = f"\nPatient {self.patient_id} - Validation Correlation per finger (Epoch {pl_module.current_epoch+1}):\n"
#             print(f"\nPatient {self.patient_id} - Validation Correlation per finger (Epoch {pl_module.current_epoch+1}):")
#             for roi in range(self.fg_num):
#                 y_hat_roi = y_prediction[:, roi]
#                 y_test_roi = y_test[:, roi]
#                 corr_tmp = corr_metric(y_hat_roi, y_test_roi)
#                 corrs.append(corr_tmp)
#                 print(f"  Finger {roi}: {corr_tmp:.4f}")
#                 metric_log += f"  Finger {roi}: {corr_tmp:.4f}\n"
#                 axi = ax.flat[roi]
#                 axi.plot(y_hat_roi, label='prediction')
#                 axi.plot(y_test_roi, label='true')
#                 axi.set_title(f"RoI {roi}_corr {corr_tmp:.2f}")
#                 axi.legend()
#
#             corr_mean = np.mean(corrs)
#             print(f"  Mean Correlation: {corr_mean:.4f}")
#             metric_log += f"  Mean Correlation: {corr_mean:.4f}\n"
#             pl_module.log("corr_mean_val", corr_mean, on_step=False, on_epoch=True, prog_bar=True)
#
#             # Save plot and metrics
#             plt.savefig(f"{RES_NPY_DIR}val_plots_p{self.patient_id}_epoch{pl_module.current_epoch+1}.png")
#             plt.close(fig)
#             with open(f"{RES_NPY_DIR}metrics_p{self.patient_id}.txt", "a") as f:
#                 f.write(metric_log)
#
# class TestCallback(Callback):
#     def __init__(self, test_x, test_y, fg_num, patient_id):
#         super().__init__()
#         self.test_x = test_x  # (channels, wavelets, time)
#         self.test_y = test_y.T  # (time, fingers)
#         self.fg_num = fg_num
#         self.patient_id = patient_id
#
#     def on_test_epoch_end(self, trainer, pl_module):
#         with torch.no_grad():
#             SIZE = 64
#             bound = self.test_x.shape[-1] // SIZE * SIZE
#             X_test = self.test_x[..., :bound]
#             print(f"TestCallback X_test shape: {X_test.shape}")
#             y_test = self.test_y[:bound, :]
#             x_batch = torch.from_numpy(X_test).float().to(pl_module.device)
#             x_batch = x_batch.unsqueeze(0)  # (1, channels, wavelets, time)
#             print(f"TestCallback x_batch shape after unsqueeze: {x_batch.shape}")
#             y_hat = pl_module.model(x_batch)[0].cpu().numpy()
#             print(f"TestCallback y_hat shape: {y_hat.shape}")
#             y_prediction = scipy.ndimage.gaussian_filter1d(y_hat.T, sigma=1).T
#             y_test = y_test[::int(DOWNSAMPLE_FS/100), :]
#
#             # Save predictions and true values
#             np.save(f"{RES_NPY_DIR}prediction_p{self.patient_id}.npy", y_prediction)
#             np.save(f"{RES_NPY_DIR}true_p{self.patient_id}.npy", y_test)
#
#             h, w = self.fg_num // 2, self.fg_num - self.fg_num // 2
#             fig, ax = plt.subplots(h, w, figsize=(h*5, w*6), sharex=True, sharey=True)
#             corrs = []
#             metric_log = f"\nPatient {self.patient_id} - Test Correlation per finger:\n"
#             print(f"\nPatient {self.patient_id} - Test Correlation per finger:")
#             for roi in range(self.fg_num):
#                 y_hat_roi = y_prediction[:, roi]
#                 y_test_roi = y_test[:, roi]
#                 corr_tmp = corr_metric(y_hat_roi, y_test_roi)
#                 corrs.append(corr_tmp)
#                 print(f"  Finger {roi}: {corr_tmp:.4f}")
#                 metric_log += f"  Finger {roi}: {corr_tmp:.4f}\n"
#                 axi = ax.flat[roi]
#                 axi.plot(y_hat_roi, label='prediction')
#                 axi.plot(y_test_roi, label='true')
#                 axi.set_title(f"RoI {roi}_corr {corr_tmp:.2f}")
#                 axi.legend()
#
#             corr_mean = np.mean(corrs)
#             print(f"  Mean Test Correlation: {corr_mean:.4f}")
#             metric_log += f"  Mean Test Correlation: {corr_mean:.4f}\n"
#             pl_module.log("corr_mean_test", corr_mean, on_step=False, on_epoch=True, prog_bar=True)
#
#             # Save plot and metrics
#             plt.savefig(f"{RES_NPY_DIR}test_plots_p{self.patient_id}.png")
#             plt.close(fig)
#             with open(f"{RES_NPY_DIR}metrics_p{self.patient_id}.txt", "a") as f:
#                 f.write(metric_log)
#
# # Main
# def main():
#     print("Starting BCI training pipeline...")
#     print("Checking CUDA availability...")
#     if not torch.cuda.is_available():
#         print("Error: CUDA not available. Please check GPU drivers and PyTorch installation.")
#         return
#     print(f"GPU detected: {torch.cuda.get_device_name(0)}")
#
#     # Load your dataset
#     print("Loading dataset...")
#     try:
#         data0 = scipy.io.loadmat("raw_training_data.mat")  # Adjust path if needed
#         if 'train_ecog' not in data0 or 'train_dg' not in data0:
#             raise KeyError("Dataset must contain 'train_ecog' and 'train_dg' keys")
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return
#     print("Dataset loaded successfully.")
#
#     # save_preprocessed_data(data0)
#
#     for patient_id in range(1, 4):
#         print(f"\n=== Training model for Patient {patient_id} ===")
#         # Model setup
#         hp_autoencoder = {
#             "channels": [32, 32, 64, 64, 128, 128],
#             "kernel_sizes": [7, 7, 5, 5, 5],
#             "strides": [2, 2, 2, 2, 2],
#             "dilation": [1, 1, 1, 1, 1],
#             "n_electrodes": CHANNELS_NUM[patient_id-1],
#             "n_freqs": WAVELET_NUM,
#             "n_channels_out": finger_num
#         }
#         model = AutoEncoder1D(**hp_autoencoder).to("cuda")
#         lightning_wrapper = BaseEcogFingerflexModel(model)
#
#         # Data
#         dm = EcogFingerflexDatamodule(sample_len=SAMPLE_LEN, patient_id=patient_id)
#         dm.setup(stage="fit")
#
#         # Validation and test data0 for callbacks
#         print(f"Loading validation and test data0 for Patient {patient_id}...")
#         try:
#             val_ecog = np.load(f"{SAVE_PATH}val/ecog_data_p{patient_id}.npy")
#             val_dg = np.load(f"{SAVE_PATH}val/fingerflex_data_p{patient_id}.npy")
#             test_ecog = np.load(f"{SAVE_PATH}test/ecog_data_p{patient_id}.npy")
#             test_dg = np.load(f"{SAVE_PATH}test/fingerflex_data_p{patient_id}.npy")
#         except FileNotFoundError as e:
#             print(f"Error: Data file not found: {e}")
#             return
#
#         # Training
#         checkpoint_callback = ModelCheckpoint(
#             save_top_k=2, monitor="corr_mean_val", mode="max",
#             dirpath=CHECKPOINT_DIR, filename=f"model_p{patient_id}-{{epoch:02d}}-{{corr_mean_val:.4f}}"
#         )
#         print(f"Starting training for Patient {patient_id}...")
#         trainer = Trainer(
#             accelerator="gpu", devices=1, max_epochs=MAX_EPOCHS,
#             callbacks=[ValidationCallback(val_ecog, val_dg, finger_num, patient_id), checkpoint_callback]
#         )
#         trainer.fit(lightning_wrapper, dm)
#         print(f"Training completed for Patient {patient_id}.")
#
#         # Testing
#         print(f"Testing model for Patient {patient_id}...")
#         dm.setup(stage="test")
#         trainer.test(lightning_wrapper, dataloaders=dm.test_dataloader(),
#                      ckpt_path="best", callbacks=[TestCallback(test_ecog, test_dg, finger_num, patient_id)])
#         print(f"Testing completed for Patient {patient_id}.")
#
# if __name__ == "__main__":
#     main()




