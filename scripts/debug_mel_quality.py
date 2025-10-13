import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Copy model classes directly
class ConvBlock(nn.Module):
    def __init__(self, ch, k=7, d=1):
        super().__init__()
        pad = (k-1)//2 * d
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, dilation=d, padding=pad)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(ch)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + y

class Wav2VecToMelTemporal(nn.Module):
    def __init__(self, in_ch=768, hidden=512, n_mels=80):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.Sequential(
            ConvBlock(hidden, k=7, d=1),
            ConvBlock(hidden, k=7, d=2),
            ConvBlock(hidden, k=7, d=4),
            ConvBlock(hidden, k=7, d=8),
            ConvBlock(hidden, k=7, d=16),
        )
        self.proj = nn.Conv1d(hidden, n_mels, 1)
    
    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.inp(X)
        H = self.blocks(H)
        M = self.proj(H)
        return M.transpose(1, 2)

# HiFi-GAN mel function
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram_hifigan(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

WAV2VEC_SR = 16000
HIFIGAN_SR = 22050
N_FFT = 1024
N_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024

def rms_norm(x, target=0.1, eps=1e-8):
    r = np.sqrt((x**2).mean() + eps)
    return x * (target / (r + eps))

# Load one test file
audio_path = "/Users/selvi.arora/Desktop/Audio_Speech_Actors_01-24 2/Actor_01/03-01-05-02-02-01-01.wav"

# Load audio
audio_orig, sr_orig = librosa.load(audio_path, sr=None)
audio_w2v = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=WAV2VEC_SR)
audio_hifi = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=HIFIGAN_SR)

# RMS norm
audio_w2v = rms_norm(audio_w2v)
audio_hifi = rms_norm(audio_hifi)

# Extract GT mel
audio_hifi_tensor = torch.FloatTensor(audio_hifi).unsqueeze(0)
mel_gt = mel_spectrogram_hifigan(audio_hifi_tensor, N_FFT, N_MELS, HIFIGAN_SR, HOP_SIZE, WIN_SIZE, 0, 8000, center=False)

# Extract wav2vec features
proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).eval()

inp = proc(audio_w2v, sampling_rate=WAV2VEC_SR, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model_w2v(**inp)
    wav2vec_feats = out.hidden_states[5].squeeze(0)

# Upsample to match mel
T_mel = mel_gt.shape[2]
wav2vec_up = torch.nn.functional.interpolate(
    wav2vec_feats.T.unsqueeze(0),
    size=T_mel,
    mode='linear',
    align_corners=False
).squeeze(0).T

# Load mapper and predict
mapper = Wav2VecToMelTemporal(in_ch=768, hidden=512, n_mels=N_MELS)
mapper.load_state_dict(torch.load('models/wav2vec_to_mel_mapper_final.pt', map_location='cpu'))
mapper.eval()

with torch.no_grad():
    mel_pred = mapper(wav2vec_up.unsqueeze(0))  # Add batch dim: [1, T, 768] -> [1, T, 80]
    mel_pred = mel_pred.squeeze(0).T.unsqueeze(0)  # [1, 80, T]

# Stats comparison
print("=" * 60)
print("MEL QUALITY DIAGNOSTIC")
print("=" * 60)

mel_gt_np = mel_gt.squeeze(0).numpy()
mel_pred_np = mel_pred.squeeze(0).numpy()

print(f"\n📊 GT MEL:")
print(f"   Shape: {mel_gt_np.shape}")
print(f"   Min: {mel_gt_np.min():.3f}, Max: {mel_gt_np.max():.3f}")
print(f"   Mean: {mel_gt_np.mean():.3f}, Std: {mel_gt_np.std():.3f}")

print(f"\n📊 PREDICTED MEL:")
print(f"   Shape: {mel_pred_np.shape}")
print(f"   Min: {mel_pred_np.min():.3f}, Max: {mel_pred_np.max():.3f}")
print(f"   Mean: {mel_pred_np.mean():.3f}, Std: {mel_pred_np.std():.3f}")

# Temporal variation check (is it flat/over-smoothed?)
gt_temporal_std = mel_gt_np.std(axis=1).mean()  # avg std over time per mel bin
pred_temporal_std = mel_pred_np.std(axis=1).mean()

print(f"\n📈 TEMPORAL VARIATION (higher = sharper):")
print(f"   GT temporal std: {gt_temporal_std:.3f}")
print(f"   Pred temporal std: {pred_temporal_std:.3f}")
print(f"   Ratio (pred/gt): {pred_temporal_std/gt_temporal_std:.3f}")

if pred_temporal_std / gt_temporal_std < 0.5:
    print("   ⚠️  PREDICTED MEL IS OVER-SMOOTHED (< 50% of GT variation)")
elif pred_temporal_std / gt_temporal_std < 0.7:
    print("   ⚠️  PREDICTED MEL IS SOMEWHAT SMOOTHED (< 70% of GT variation)")
else:
    print("   ✅ PREDICTED MEL HAS GOOD TEMPORAL DETAIL")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].imshow(mel_gt_np, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title('Ground Truth Mel')
axes[0].set_ylabel('Mel Bin')

axes[1].imshow(mel_pred_np, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title('Predicted Mel')
axes[1].set_ylabel('Mel Bin')
axes[1].set_xlabel('Time Frame')

plt.tight_layout()
plt.savefig('mel_comparison.png', dpi=150)
print(f"\n💾 Saved visualization: mel_comparison.png")
print("\nLook for: GT should have sharp vertical lines (consonants), Pred should too!")

