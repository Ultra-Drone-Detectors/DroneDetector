import numpy as np
import librosa
import torch
import os
from train_cnn import DroneCNN   # dùng model class cho 4 lớp


# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/drone_cnn_4class.pth"
DATA_MEL_DIR = "data_melspec"   

SAMPLE_RATE = 22050
WINDOW_SIZE = 2.0                # mỗi cửa sổ 1 giây
HOP_SIZE = 1                   # trượt 0.5 giây = 50% overlap
CONF_THRESHOLD = 0.50            # reject nếu confidence < 60%


# ==============================
# LOAD MODEL + CLASS NAMES
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = sorted([
    c for c in os.listdir(DATA_MEL_DIR)
    if os.path.isdir(os.path.join(DATA_MEL_DIR, c))
])

num_classes = len(CLASS_NAMES)

model = DroneCNN(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Classes:", CLASS_NAMES)
print("Model loaded.\n")


# ==============================
# MEL SPECTROGRAM
# ==============================
def make_melspec(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # normalize per mel
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel


# ==============================
# PREDICT SINGLE WINDOW
# ==============================
def predict_segment(y_segment):
    mel = make_melspec(y_segment)
    mel_tensor = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(mel_tensor)
        prob = torch.softmax(pred, dim=1)

    conf, cls = torch.max(prob, dim=1)
    return CLASS_NAMES[cls.item()], conf.item()


# ==============================
# SLIDING WINDOW DETECTION
# ==============================
def sliding_window_detect(filepath):
    print(f"🔊 Processing long file: {filepath}\n")

    # load full audio
    y, sr = librosa.load(filepath, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    window_len = int(WINDOW_SIZE * SAMPLE_RATE)
    hop_len = int(HOP_SIZE * SAMPLE_RATE)

    timeline = []

    for start in range(0, len(y) - window_len, hop_len):
        end = start + window_len
        y_seg = y[start:end]

        cls, conf = predict_segment(y_seg)

        # convert sample index → seconds
        t_start = start / SAMPLE_RATE
        t_end = end / SAMPLE_RATE

        # reject if confidence too low
        if conf < CONF_THRESHOLD:
            timeline.append((t_start, t_end, "unknown", conf))
        else:
            timeline.append((t_start, t_end, cls, conf))

    return timeline


# ==============================
# DISPLAY TIMELINE
# ==============================
def print_timeline(timeline):
    print("\n📌 Sliding Window Results:")
    for t_start, t_end, cls, conf in timeline:
        print(f"{t_start:5.2f}–{t_end:5.2f}s → {cls:10s}   conf={conf:.2f}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    test_file = "test_audio/droneI/I_05.wav"   
    timeline = sliding_window_detect(test_file)
    print_timeline(timeline)
