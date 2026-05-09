import base64
import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "mitbih_arrhythmia_cnn.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def stream(name, text):
    return {"name": name, "output_type": "stream", "text": text.splitlines(keepends=True)}


def png(data):
    return {
        "data": {"image/png": base64.b64encode(data).decode("ascii"), "text/plain": ["<Figure>"]},
        "metadata": {},
        "output_type": "display_data",
    }


def execute(cells):
    ns = {"__name__": "__main__"}
    count = 1
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        out = io.StringIO()
        err = io.StringIO()
        src = "".join(cell["source"])
        try:
            with redirect_stdout(out), redirect_stderr(err):
                exec(compile(src, f"cell_{count}", "exec"), ns)
        except Exception:
            tb = traceback.format_exc()
            cell["execution_count"] = count
            cell["outputs"] = [{
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": "Notebook generation failed",
                "traceback": tb.splitlines(),
            }]
            raise
        outputs = []
        if out.getvalue():
            outputs.append(stream("stdout", out.getvalue()))
        if err.getvalue():
            outputs.append(stream("stderr", err.getvalue()))
        try:
            import matplotlib.pyplot as plt

            for num in plt.get_fignums():
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
                outputs.append(png(buf.getvalue()))
            plt.close("all")
        except Exception:
            pass
        cell["execution_count"] = count
        cell["outputs"] = outputs
        count += 1


cells = [
    md(
        """# MIT-BIH: arytmia vs brak arytmii przy uzyciu 1D CNN

## Co robimy

Rozwiazujemy binarny problem klasyfikacji fragmentow EKG:
- `0` - brak arytmii
- `1` - arytmia

Najwazniejsze zalozenia:
- korzystamy z danych MIT-BIH Arrhythmia Database
- sygnal dzielimy na krotkie segmenty 1D wokol adnotowanych pobudzen
- walidacja jest grupowa po pacjentach, bez mieszania osob miedzy treningiem i testem
- model to prosta siec CNN 1D

Uwaga o pacjentach:
- MIT-BIH ma 48 rekordow od 47 osob
- rekordy `201` i `202` traktuja o tym samym pacjencie, wiec sa laczone do jednej grupy
"""
    ),
    code(
        """import math
import os
import random
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_mitbih"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import GroupKFold

plt.style.use("seaborn-v0_8-whitegrid")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("MitBih/mitbih_database")
FS = 360
SEGMENT_LEN = 256
HALF = SEGMENT_LEN // 2
MAX_PER_CLASS_PER_PATIENT = 80
DEVICE = "cpu"

BEAT_SYMBOLS = {"N", "L", "R", "e", "j", "A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"}
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}

print("Seed:", SEED)
print("Segment length:", SEGMENT_LEN, "samples")
print("Records:", len(list(DATA_DIR.glob("*.csv"))))
print("Device:", DEVICE)
"""
    ),
    code(
        """def patient_id(record_id):
    return "201_202" if record_id in {"201", "202"} else record_id


def load_signal(record_id):
    x = pd.read_csv(DATA_DIR / f"{record_id}.csv", usecols=[1]).iloc[:, 0].to_numpy(dtype=np.float32)
    b, a = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    x = filtfilt(b, a, x).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x


def load_annotations(record_id):
    samples, symbols = [], []
    with open(DATA_DIR / f"{record_id}annotations.txt") as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            sample = int(parts[1])
            symbol = parts[2]
            if symbol in BEAT_SYMBOLS:
                samples.append(sample)
                symbols.append(symbol)
    return np.array(samples), np.array(symbols)


def build_dataset(max_per_class=MAX_PER_CLASS_PER_PATIENT):
    records = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    X, y, groups, rows = [], [], [], []
    rng = np.random.default_rng(SEED)

    for record in records:
        signal = load_signal(record)
        samples, symbols = load_annotations(record)
        valid = (samples >= HALF) & (samples + HALF < len(signal))
        samples, symbols = samples[valid], symbols[valid]
        labels = np.array([0 if s in NORMAL_SYMBOLS else 1 for s in symbols], dtype=np.int64)

        chosen = []
        for cls in [0, 1]:
            idx = np.where(labels == cls)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            chosen.extend(idx.tolist())
        chosen = np.array(sorted(chosen, key=lambda i: samples[i]))

        pid = patient_id(record)
        for i in chosen:
            center = samples[i]
            seg = signal[center - HALF:center + HALF]
            label = int(labels[i])
            X.append(seg)
            y.append(label)
            groups.append(pid)
            rows.append({"record": record, "patient": pid, "label": label, "symbol": symbols[i]})

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    info = pd.DataFrame(rows)
    return X, y, groups, info


X, y, groups, info = build_dataset()
patient_stats = info.groupby("patient").agg(
    segments=("label", "size"),
    arrhythmia=("label", "sum"),
).reset_index()
patient_stats["normal"] = patient_stats["segments"] - patient_stats["arrhythmia"]
patient_stats["arrhythmia_ratio"] = patient_stats["arrhythmia"] / patient_stats["segments"]

print("Dataset shape:", X.shape)
print("Patients:", len(np.unique(groups)))
print("Normal:", int((y == 0).sum()), "Arrhythmia:", int((y == 1).sum()))
print(patient_stats.sort_values("segments", ascending=False).head(10).round(3).to_string(index=False))
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

class_counts = pd.Series({"Brak arytmii": int((y == 0).sum()), "Arytmia": int((y == 1).sum())})
axes[0].bar(class_counts.index, class_counts.values, color=["#2563eb", "#dc2626"], width=0.6)
axes[0].set_title("Bilans klas")
axes[0].set_ylabel("Liczba segmentow")

axes[1].hist(patient_stats["segments"], bins=10, color="#0f766e", edgecolor="white")
axes[1].set_title("Segmenty na pacjenta")
axes[1].set_xlabel("Liczba segmentow")

top_patients = patient_stats.sort_values("segments", ascending=False).head(12).iloc[::-1]
axes[2].barh(top_patients["patient"], top_patients["arrhythmia_ratio"], color="#f59e0b")
axes[2].set_title("Udzial arytmii w najwiekszych grupach")
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Odsetek arytmii")
plt.tight_layout()

sample_record = "106"
signal = load_signal(sample_record)
samples, symbols = load_annotations(sample_record)
valid = (samples >= HALF) & (samples + HALF < len(signal))
samples, symbols = samples[valid], symbols[valid]
normal_i = next(i for i, s in enumerate(symbols) if s in NORMAL_SYMBOLS)
arr_i = next(i for i, s in enumerate(symbols) if s not in NORMAL_SYMBOLS)

t = (np.arange(SEGMENT_LEN) - HALF) / FS
normal_seg = signal[samples[normal_i] - HALF:samples[normal_i] + HALF]
arr_seg = signal[samples[arr_i] - HALF:samples[arr_i] + HALF]

plt.figure(figsize=(12, 4))
plt.plot(t, normal_seg, label="Brak arytmii", linewidth=2, color="#2563eb")
plt.plot(t, arr_seg, label="Arytmia", linewidth=2, color="#dc2626")
plt.axvline(0, linestyle="--", color="black", alpha=0.6)
plt.title("Przykladowe segmenty EKG wokol pobudzenia")
plt.xlabel("Czas wzgledem adnotacji [s]")
plt.ylabel("Znormalizowana amplituda")
plt.legend()
plt.tight_layout()
"""
    ),
    code(
        """class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(16, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_tensors(X_np, y_np):
    X_t = torch.tensor(X_np[:, None, :], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_np[:, None], dtype=torch.float32, device=DEVICE)
    return X_t, y_t


def train_fold(X_train, y_train, X_val, y_val, epochs=8, batch_size=64):
    model = SimpleCNN().to(DEVICE)
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train_t, y_train_t = make_tensors(X_train, y_train)
    X_val_t, y_val_t = make_tensors(X_val, y_val)
    history = []
    best_state, best_val = None, float("inf")

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_train_t))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = model(X_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train_t), y_train_t).item()
            val_loss = criterion(model(X_val_t), y_val_t).item()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_proba(model, X_np):
    model.eval()
    X_t, _ = make_tensors(X_np, np.zeros(len(X_np), dtype=np.int64))
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy().ravel()
    return probs


def fold_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc_abs": abs(matthews_corrcoef(y_true, y_pred)),
    }
"""
    ),
    code(
        """outer_cv = GroupKFold(n_splits=3)
results, histories = [], []
all_true, all_prob = [], []
split_rows = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    train_groups = groups[train_idx]
    inner_groups = np.unique(train_groups)
    inner_cv = GroupKFold(n_splits=min(4, len(inner_groups)))
    inner_train_rel, val_rel = next(inner_cv.split(X[train_idx], y[train_idx], train_groups))
    inner_train_idx = train_idx[inner_train_rel]
    val_idx = train_idx[val_rel]

    assert len(set(groups[inner_train_idx]) & set(groups[val_idx])) == 0
    assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0

    model, hist = train_fold(X[inner_train_idx], y[inner_train_idx], X[val_idx], y[val_idx])
    prob = predict_proba(model, X[test_idx])
    metrics = fold_metrics(y[test_idx], prob)
    metrics["fold"] = fold
    results.append(metrics)
    hist["fold"] = fold
    histories.append(hist)
    all_true.append(y[test_idx])
    all_prob.append(prob)
    split_rows.append({
        "fold": fold,
        "test_segments": len(test_idx),
        "test_patients": len(np.unique(groups[test_idx])),
        "test_arrhythmia": int(y[test_idx].sum()),
        "test_normal": int((y[test_idx] == 0).sum()),
    })

results_df = pd.DataFrame(results).sort_values("fold").reset_index(drop=True)
history_df = pd.concat(histories, ignore_index=True)
split_df = pd.DataFrame(split_rows)
mean_df = results_df[["accuracy", "precision", "recall", "f1_score", "mcc_abs"]].mean().to_frame().T

print("Foldy testowe:")
print(split_df.to_string(index=False))
print("\\nMetryki na foldach:")
print(results_df.round(4).to_string(index=False))
print("\\nSrednie metryki:")
print(mean_df.round(4).to_string(index=False))
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for fold, hist in history_df.groupby("fold"):
    axes[0].plot(hist["epoch"], hist["val_loss"], marker="o", linewidth=1.8, label=f"Fold {fold}")
axes[0].set_title("Walidacyjna funkcja straty")
axes[0].set_xlabel("Epoka")
axes[0].set_ylabel("Loss")
axes[0].legend()

x = np.arange(5)
metric_cols = ["accuracy", "precision", "recall", "f1_score", "mcc_abs"]
for i, row in results_df.iterrows():
    axes[1].bar(x + (i - 1) * 0.22, row[metric_cols], width=0.22, label=f"Fold {int(row['fold'])}")
axes[1].plot(x, mean_df.iloc[0].values, color="black", marker="o", linewidth=2.0, label="Srednia")
axes[1].set_xticks(x, ["accuracy", "precision", "recall", "f1", "|MCC|"])
axes[1].set_ylim(0, 1)
axes[1].set_title("Porownanie metryk")
axes[1].legend()
plt.tight_layout()

y_true = np.concatenate(all_true)
y_prob = np.concatenate(all_prob)
y_pred = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(4.8, 4.2))
im = plt.imshow(cm, cmap="YlOrRd")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Macierz pomylek")
for i in range(2):
    for j in range(2):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
"""
    ),
    md(
        """## Wnioski

- Notebook zachowuje najwazniejsze elementy zadania: segmentacje EKG, grupowy podzial po pacjentach, CNN 1D i metryki z walidacji krzyzowej.
- Najwieksze ryzyko przecieku danych zostalo ograniczone przez `GroupKFold`.
- Model jest prosty, ale wystarczajacy do pokazania pelnego pipeline'u klasyfikacji `arytmia` vs `brak arytmii`.
- Dalsze ulepszenia to strojenie hiperparametrow, glebsza architektura oraz dokladniejsza segmentacja wokol pikow R.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


execute(notebook["cells"])
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2))
print(f"Saved notebook to: {NOTEBOOK_PATH}")
