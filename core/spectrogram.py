import io
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
import matplotlib.cm as cm
from PIL import Image

def generate_mel_spectrogram(
    y,
    sr=44100,
    fmin=200,
    fmax=15000,
    n_mels=224,
    n_fft=1024,
    hop_length=256,
    target_size=(224, 224)
):
    """
    Standardized mel spectrogram generator for training, inference, and visualization.
    Renders pure acoustic data with no titles, axes, borders, or text artifacts.
    """
    # 4th-order highpass filter at 200 Hz to suppress vessel engine noise
    if sr > 400:
        sos = butter(4, 200, btype='highpass', fs=sr, output='sos')
        y_filt = sosfiltfilt(sos, y)
    else:
        y_filt = y

    # Pre-emphasis to boost higher frequency harmonics
    y_emph = librosa.effects.preemphasis(y_filt, coef=0.97)

    # Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y_emph,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    S_dB = librosa.power_to_db(S, ref=np.max)
    denom = S_dB.max() - S_dB.min()
    S_norm = (S_dB - S_dB.min()) / (denom if denom > 1e-6 else 1.0)

    # Map to RGB using magma colormap without matplotlib axis/text overhead
    colored = cm.magma(S_norm)[:, :, :3]
    # Place lower frequencies at the bottom
    colored = np.flipud(colored)
    img_uint8 = (colored * 255).astype(np.uint8)

    img = Image.fromarray(img_uint8)
    if target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    return img, S_norm

def audio_path_to_spectrogram(audio_path, duration=2.0, sr=44100, target_size=(224, 224)):
    y, actual_sr = librosa.load(audio_path, sr=sr, duration=duration)
    return generate_mel_spectrogram(y, sr=actual_sr, target_size=target_size)

def save_spectrogram(audio_path, output_path, duration=2.0, sr=44100):
    img, _ = audio_path_to_spectrogram(audio_path, duration=duration, sr=sr)
    img.save(output_path)
