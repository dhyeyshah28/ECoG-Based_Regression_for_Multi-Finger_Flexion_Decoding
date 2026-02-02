import numpy as np
import scipy.io
import scipy.signal
import pywt
import torch
from collections import OrderedDict
import glob
from bci_notebook import AutoEncoder1D, CHANNELS_NUM  # Assuming your model is in model.py

# Fixed file paths
INPUT_MAT_PATH = "C:\\Users\\dhyey\\OneDrive\\Desktop\\MSE_ROBO\\Sem_2\\Brain_Computer_Interfaces\\Assignments\\Final_Project\\leaderboard_data.mat"
OUTPUT_MAT_PATH = "predictions_final_normal5_200same_0.430_0.425_0.610_0.384.mat"
CHECKPOINT_DIR = "C:\\Users\\dhyey\\OneDrive\\Desktop\\MSE_ROBO\\Sem_2\\Brain_Computer_Interfaces\\Assignments\\checkpoints\\"

# Constants
WAVELET_NUM = 40
L_FREQ, H_FREQ = 5, 200
ORIGINAL_FS = 1000
DOWNSAMPLE_FS = 100
CHANNELS_NUM = [62, 48, 64]


class LiveECoGInference:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.downsample_factor = ORIGINAL_FS // DOWNSAMPLE_FS

    def load_model(self):
        checkpoint_path = f"{CHECKPOINT_DIR}model_p{self.patient_id}-*-val_corr=*.ckpt"
        checkpoints = glob.glob(checkpoint_path)
        if not checkpoints:
            checkpoint_path = f"{CHECKPOINT_DIR}model_p{self.patient_id}-*-corr_mean_val=*.ckpt"
            checkpoints = glob.glob(checkpoint_path)
        checkpoint_path = max(checkpoints) if checkpoints else None

        print(f"Loading model for Patient {self.patient_id} from: {checkpoint_path}")

        model = AutoEncoder1D(
            n_electrodes=CHANNELS_NUM[self.patient_id - 1],
            n_freqs=WAVELET_NUM,
            n_channels_out=5
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        fixed_state_dict = OrderedDict((k.replace("model.", ""), v) for k, v in state_dict.items())

        model.load_state_dict(fixed_state_dict)
        model.eval()
        return model

    def preprocess(self, ecog_data):
        # Bandpass filter with safe padding
        sos = scipy.signal.butter(4, [L_FREQ, H_FREQ], btype='band', fs=ORIGINAL_FS, output='sos')
        padlen = min(27, ecog_data.shape[0] // 3)  # Conservative padding
        ecog_filtered = scipy.signal.sosfiltfilt(sos, ecog_data, axis=0, padlen=padlen)

        # Wavelet transform
        scales = pywt.scale2frequency('cmor1.5-1.0', np.arange(1, WAVELET_NUM + 1)) * ORIGINAL_FS
        scales = scales[(scales >= L_FREQ) & (scales <= H_FREQ)]

        ecog_wavelet = np.zeros((ecog_data.shape[1], WAVELET_NUM, ecog_data.shape[0]))
        for ch in range(ecog_data.shape[1]):
            coef, _ = pywt.cwt(ecog_filtered[:, ch], scales, 'cmor1.5-1.0', sampling_period=1 / ORIGINAL_FS)
            ecog_wavelet[ch, :coef.shape[0], :] = np.abs(coef)

        # Downsample carefully
        return scipy.signal.decimate(ecog_wavelet, self.downsample_factor, axis=-1).astype(np.float32)

    def predict(self, ecog_data):
        original_length = ecog_data.shape[0]

        # Preprocess to 100Hz (147500 → 14750 samples)
        processed_ecog = self.preprocess(ecog_data)

        # Window parameters
        window_size = 256
        stride = 1

        # Calculate needed padding
        n_windows = (processed_ecog.shape[2] - window_size) // stride + 1
        needed_length = (n_windows - 1) * stride + window_size
        current_length = processed_ecog.shape[2]

        # Only pad if needed and with positive padding values
        if needed_length > current_length:
            padding = needed_length - current_length
            processed_ecog = np.pad(processed_ecog,
                                    ((0, 0), (0, 0), (0, max(0, padding))),
                                    mode='edge')

        # Create windows
        n_windows = (processed_ecog.shape[2] - window_size) // stride + 1
        windows = np.zeros((n_windows, processed_ecog.shape[0], processed_ecog.shape[1], window_size))
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            windows[i] = processed_ecog[:, :, start:end]

        # Predict in batches
        predictions = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, n_windows, batch_size):
                batch = torch.from_numpy(windows[i:i + batch_size]).float().to(self.device)
                predictions.append(self.model(batch).cpu().numpy())

        # Stitch windows with overlap-add
        output_length = (n_windows - 1) * stride + window_size
        continuous_pred = np.zeros((5, output_length))
        counts = np.zeros(output_length)

        for i, pred in enumerate(np.concatenate(predictions, axis=0)):
            start = i * stride
            end = start + window_size
            continuous_pred[:, start:end] += pred
            counts[start:end] += 1

        continuous_pred /= np.maximum(counts, 1)

        # Upsample to 1000Hz (14750 → 147500 samples)
        upsampled = scipy.signal.resample(
            continuous_pred,
            original_length,
            axis=1
        )

        return upsampled


def main():
    # Load input data0
    data = scipy.io.loadmat(INPUT_MAT_PATH)
    leaderboard_ecog = data['leaderboard_ecog']  # Shape: (3, 1) cell array

    # Process each patient
    predicted_dg = np.zeros((3, 1), dtype=object)

    for patient_id in [1, 2, 3]:
        print(f"\nProcessing Patient {patient_id}...")

        # Get patient data0 (N x channels)
        ecog_data = leaderboard_ecog[patient_id - 1, 0]  # Access the MATLAB cell

        print(f"Input shape: {ecog_data.shape} (samples × channels)")
        print(f"Duration: {ecog_data.shape[0] / ORIGINAL_FS:.2f} seconds")

        # Run inference
        predictions = LiveECoGInference(patient_id).predict(ecog_data)
        predicted_dg[patient_id - 1, 0] = predictions.T  # Store as [samples, 5]

        print(f"Output shape: {predictions.T.shape}")

    # Save results0
    scipy.io.savemat(OUTPUT_MAT_PATH, {'predicted_dg': predicted_dg})
    print(f"\nPredictions saved to {OUTPUT_MAT_PATH}")
    print("Final output structure:")
    for i in range(3):
        print(f"Patient {i + 1}: {predicted_dg[i, 0].shape} (samples × 5)")


if __name__ == "__main__":
    main()