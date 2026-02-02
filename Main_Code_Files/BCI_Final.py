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
WAVELET_NUM = 40          # Number of wavelets
DOWNSAMPLE_FS = 100       # Target sampling rate
time_delay_secs = 0.187    # Time delay
SAMPLE_LEN = 256          # Window size
finger_num = 5            # Number of fingers
ORIGINAL_FS = 1000        # Assumed original sampling rate
BATCH_SIZE = 64           # Adjusted for RTX 4070
CHANNELS_NUM = [62, 48, 64]  # Channels per patient
MAX_EPOCHS = 5           # Training epochs


# Paths
SAVE_PATH = f"{pathlib.Path().resolve()}/data0/"
CHECKPOINT_DIR = f"{pathlib.Path().resolve()}/checkpoints/"
RES_NPY_DIR = f"{pathlib.Path().resolve()}/results/"
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
    """Downsample fingerflex data."""
    print("  Downsampling fingerflex data...")
    downsample_factor = int(original_fs // target_fs)
    fingerflex_downsampled = scipy.signal.decimate(fingerflex_data, downsample_factor, axis=0)
    return fingerflex_downsampled.astype(np.float32)

def save_preprocessed_data(data):
    """Preprocess and save data for each patient."""
    print("Preprocessing data for all patients...")
    for p in range(3):
        print(f"Processing Patient {p + 1}...")
        ecog = data['train_ecog'][p, 0]
        dg = data['train_dg'][p, 0]

        ecog_proc = preprocess_ecog(ecog)
        dg_proc = preprocess_fingerflex(dg)

        total_time = ecog_proc.shape[-1]
        train_idx = int(0.7 * total_time)
        val_idx = int(0.85 * total_time)

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
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        if self.test is None:
            self.setup(stage="test")  # Ensure test dataset is initialized
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, pin_memory=True)

class WaveletInitializedConv1d(nn.Module):
    """1D convolution layer initialized with wavelet filters"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, dilation=dilation)
        
        # Initialize with wavelet filters if possible
        if kernel_size % 2 == 1 and out_channels >= 2:
            try:
                wavelet = pywt.Wavelet('bior6.8')
                dec_lo, dec_hi = wavelet.dec_lo, wavelet.dec_hi
                
                # Pad wavelet filters to match kernel size
                pad_len = (kernel_size - len(dec_lo)) // 2
                dec_lo = np.pad(dec_lo, (pad_len, pad_len), 'constant')
                dec_hi = np.pad(dec_hi, (pad_len, pad_len), 'constant')
                
                with torch.no_grad():
                    self.conv.weight[0] = torch.tensor(dec_lo, dtype=torch.float32)
                    self.conv.weight[1] = torch.tensor(dec_hi, dtype=torch.float32)
                    if out_channels > 2:
                        nn.init.xavier_uniform_(self.conv.weight[2:])
            except:
                print("Warning: Could not initialize with wavelet filters")
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            nn.init.xavier_uniform_(self.conv.weight)
            
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, p_conv_drop=0.1):
        super().__init__()
        self.conv1d = WaveletInitializedConv1d(in_channels, out_channels, kernel_size, 
                                             stride=1, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
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
                 strides=[2, 2, 2, 2, 2], dilation=[1, 2, 4, 8, 1]):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs * n_electrodes
        self.model_depth = len(channels) - 1
        
        # Spatial convolution for channel mixing
        self.spatial_conv = nn.Conv2d(n_electrodes, n_electrodes, 
                                    kernel_size=(1, 1), padding='same')
        
        # Initial spatial reduction
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
        
        # Downsample blocks with increasing dilation
        self.downsample_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], kernel_sizes[i], 
                     strides[i], dilation[i])
            for i in range(self.model_depth)
        ])
        
        # Upsample blocks
        channels = channels[:-1] + channels[-1:]
        self.upsample_blocks = nn.ModuleList([
            UpConvBlock(scale=strides[i],
                       in_channels=channels[i + 1] if i == self.model_depth - 1 else channels[i + 1] * 2,
                       out_channels=channels[i], kernel_size=kernel_sizes[i])
            for i in range(self.model_depth - 1, -1, -1)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=channels[0] * 2, hidden_size=64, 
                           batch_first=True, bidirectional=True)
        
        # Final output layers
        self.conv1x1_one = nn.Conv1d(128, n_channels_out, kernel_size=1, padding='same')
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize spatial conv with identity
        if hasattr(self, 'spatial_conv'):
            nn.init.dirac_(self.spatial_conv.weight)
            nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, x):
        batch, elec, n_freq, time = x.shape
        
        # Spatial convolution across channels
        x = self.spatial_conv(x)
        
        # Reshape for temporal processing
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        # Downsample with skip connections
        skip_connection = []
        for i in range(self.model_depth):
            skip_connection.append(x)
            x = self.downsample_blocks[i](x)
        
        # Upsample with skip connections
        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat((x, skip_connection[-1 - i]), dim=1)
        
        # LSTM temporal modeling
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # (batch, features, time)
        
        # Final output
        x = self.conv1x1_one(x)
        return x

class BaseEcogFingerflexModel(pl.LightningModule):
    def __init__(self, model, patient_id):
        super().__init__()
        self.model = model
        self.patient_id = patient_id
        self.lr = 8.42e-5
        self.all_metrics = {
            'train': {f'finger_{i}': [] for i in range(finger_num)},
            'val': {f'finger_{i}': [] for i in range(finger_num)},
            'test': {f'finger_{i}': [] for i in range(finger_num)}
        }
        self.save_hyperparameters()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_corr',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def correlation_metric(x, y):
    x = x.flatten()
    y = y.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr

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

    # Load and preprocess data
    try:
        data = scipy.io.loadmat("C:\\Users\\dhyey\\OneDrive\\Desktop\\MSE_ROBO\\Sem_2\\Brain_Computer_Interfaces\\Assignments\\Final_Project\\raw_training_data.mat")
        if 'train_ecog' not in data or 'train_dg' not in data:
            raise KeyError("Dataset must contain 'train_ecog' and 'train_dg' keys")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Save preprocessed data if not already exists
    if not os.path.exists(f"{SAVE_PATH}train/ecog_data_p1.npy"):
        print("Preprocessing data...")
        save_preprocessed_data(data)
    else:
        print("Preprocessed data already exists, skipping preprocessing...")

    all_patient_results = {}
    for patient_id in range(1, 4):
        print(f"\n=== Processing Patient {patient_id} ===")

        # Initialize model with patient-specific parameters
        hp_autoencoder = {
            "channels": [32, 32, 64, 64, 128, 128],
            "kernel_sizes": [7, 7, 5, 5, 5],
            "strides": [2, 2, 2, 2, 2],
            "dilation": [1, 2, 4, 8, 1],  # Increased dilation rates
            "n_electrodes": CHANNELS_NUM[patient_id - 1],
            "n_freqs": WAVELET_NUM,
            "n_channels_out": finger_num
        }
        model = AutoEncoder1D(**hp_autoencoder).to("cuda")
        lightning_wrapper = BaseEcogFingerflexModel(model, patient_id)

        # Initialize data module
        dm = EcogFingerflexDatamodule(
            sample_len=SAMPLE_LEN,
            patient_id=patient_id
        )

        # Setup data - explicitly for training first
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
            dm.setup(stage="test")  # Explicitly setup test data
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

        # Store results
        all_patient_results[f"patient_{patient_id}"] = lightning_wrapper.all_metrics

    # Generate final comparison plots
    print("\nGenerating final comparison plots...")
    plot_final_comparison(all_patient_results)
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()