from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "ecg_wavelet_layers_cnn_arrhythmia.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (.venv Lab8)",
        "language": "python",
        "name": "lab8-ecg-wavelet-cnn",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nb["cells"] = [
    md(
        """
        # Klasyfikacja ramek EKG: konwolucyjne warstwy falkowe

        Notebook wykonuje klasyfikację ramek sygnału EKG z MIT-BIH na dwie klasy:
        sygnał bez arytmii i sygnał z arytmią. Styl przygotowania danych i walidacji
        jest zgodny z Lab7: segmenty są wycinane z lokalnych plików MIT-BIH, dane są
        balansowane per rekord, a walidacja pilnuje, aby rekord/pacjent z części
        testowej nie pojawiał się w treningu.

        Różnica względem Lab7 jest celowa: wejściem sieci są ramki 1D EKG, a nie obrazy
        CWT. Pierwsze warstwy są warstwami falkowymi, czyli stałymi konwolucjami 1D
        filtrami dekompozycji falkowej. Porównane konfiguracje:

        - jedna warstwa falkowa,
        - dwie warstwy falkowe,
        - jedna warstwa falkowa i jedna zwykła warstwa konwolucyjna.
        """
    ),
    code(
        """
        from pathlib import Path
        import re
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import pywt

        from IPython.display import display
        from scipy.signal import correlate
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)

        ROOT = Path.cwd()
        DATA_DIR = ROOT / "MIT-BIH" / "mitbih_database"
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Brak katalogu danych: {DATA_DIR}")

        print(f"Katalog danych: {DATA_DIR}")
        print(f"Liczba plików CSV: {len(list(DATA_DIR.glob('*.csv')))}")
        """
    ),
    md(
        """
        ## Przygotowanie ramek EKG

        MIT-BIH ma sygnały próbkowane z częstotliwością 360 Hz. Ramki są wycinane wokół
        adnotowanych pobudzeń. Klasa `0` oznacza pobudzenia normalne według grupy AAMI
        (`N`, `L`, `R`, `e`, `j`), a klasa `1` pozostałe pobudzenia traktowane jako
        arytmia. Artefakty i adnotacje techniczne są pomijane.

        Z każdego rekordu pobierana jest ograniczona, zbalansowana liczba ramek, aby
        notebook liczył się szybko i aby rekordy z wieloma normalnymi pobudzeniami nie
        zdominowały treningu.
        """
    ),
    code(
        """
        FS = 360
        WINDOW = 256
        HALF_WINDOW = WINDOW // 2
        NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
        EXCLUDED_SYMBOLS = {"+", "~", "|", "!", "[", "]", "x", "f", "Q", "?", '"'}
        MAX_PER_CLASS_PER_RECORD = 24


        def read_annotations(path: Path) -> pd.DataFrame:
            rows = []
            pattern = re.compile(r"^\\s*(\\S+)\\s+(\\d+)\\s+(\\S+)\\s+")
            for line in path.read_text(errors="ignore").splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                _, sample, beat_type = match.groups()
                if beat_type in EXCLUDED_SYMBOLS:
                    continue
                rows.append({"sample": int(sample), "type": beat_type})
            return pd.DataFrame(rows)


        def zscore(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32)
            return (x - x.mean()) / (x.std() + 1e-8)


        def load_frames(data_dir: Path) -> pd.DataFrame:
            records = []
            for csv_path in sorted(data_dir.glob("*.csv")):
                record_id = csv_path.stem
                ann_path = data_dir / f"{record_id}annotations.txt"
                if not ann_path.exists():
                    continue

                signal_df = pd.read_csv(csv_path)
                signal_df.columns = [str(col).strip().strip("'\\\"") for col in signal_df.columns]
                channel = "MLII" if "MLII" in signal_df.columns else [col for col in signal_df.columns if col != "sample #"][0]
                signal = signal_df[channel].to_numpy(dtype=np.float32)
                annotations = read_annotations(ann_path)

                local = []
                for _, row in annotations.iterrows():
                    center = int(row["sample"])
                    start = center - HALF_WINDOW
                    end = center + HALF_WINDOW
                    if start < 0 or end > len(signal):
                        continue
                    label = 0 if row["type"] in NORMAL_SYMBOLS else 1
                    local.append(
                        {
                            "record": record_id,
                            "sample": center,
                            "symbol": row["type"],
                            "label": label,
                            "frame": zscore(signal[start:end]),
                        }
                    )

                if not local:
                    continue
                local_df = pd.DataFrame(local)
                balanced_parts = []
                rng = np.random.default_rng(RANDOM_STATE + int(record_id))
                for label, group in local_df.groupby("label"):
                    take = min(MAX_PER_CLASS_PER_RECORD, len(group))
                    balanced_parts.append(group.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
                records.append(pd.concat(balanced_parts, ignore_index=True))

            if not records:
                raise RuntimeError("Nie znaleziono ramek EKG do analizy.")
            return pd.concat(records, ignore_index=True)


        frames = load_frames(DATA_DIR)
        X_frames = np.stack(frames["frame"].to_numpy()).astype(np.float32)
        y = frames["label"].to_numpy(dtype=np.int64)
        groups = frames["record"].to_numpy()

        display(frames.drop(columns=["frame"]).head())
        print("Liczba ramek:", len(frames))
        print("Liczba rekordów/pacjentów:", frames["record"].nunique())
        display(pd.crosstab(frames["record"], frames["label"], margins=True).rename(columns={0: "normal", 1: "arytmia"}))
        """
    ),
    code(
        """
        label_names = {0: "bez arytmii", 1: "arytmia"}
        label_counts = frames["label"].map(label_names).value_counts().reindex(["bez arytmii", "arytmia"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
        axes[0].bar(label_counts.index, label_counts.values, color=["#2f6f9f", "#c44e52"])
        axes[0].set_title("Liczba ramek po balansowaniu")
        axes[0].set_ylabel("liczba ramek")

        per_record = pd.crosstab(frames["record"], frames["label"]).rename(columns=label_names)
        per_record.plot(kind="bar", stacked=True, ax=axes[1], color=["#2f6f9f", "#c44e52"], width=0.85)
        axes[1].set_title("Ramki per rekord/pacjent")
        axes[1].set_xlabel("rekord")
        axes[1].set_ylabel("liczba ramek")
        axes[1].tick_params(axis="x", labelsize=7)
        axes[1].legend(title="klasa")
        plt.show()
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 5), constrained_layout=True, sharex=True)
        time_axis = np.arange(WINDOW) / FS
        for row, label in enumerate([0, 1]):
            examples = frames[frames["label"] == label].sort_values(["record", "sample"]).head(2)
            for col, (_, example) in enumerate(examples.iterrows()):
                axes[row, col].plot(time_axis, example["frame"], linewidth=1.1)
                axes[row, col].set_title(
                    f"{label_names[label]} | rekord {example['record']} | symbol {example['symbol']}"
                )
                axes[row, col].set_xlabel("czas [s]")
                axes[row, col].set_ylabel("amplituda z-score")
                axes[row, col].grid(alpha=0.25)
        plt.show()
        """
    ),
    md(
        """
        ## Warstwy falkowe jako konwolucje 1D

        Warstwa falkowa używa filtrów dekompozycji wybranej falki (`db4`): dolnoprzepustowego
        i górnoprzepustowego. Dla każdej ramki wykonywana jest konwolucja, aktywacja ReLU
        i redukcja długości przez próbkowanie co drugi punkt. Druga warstwa falkowa działa
        analogicznie na mapach cech z warstwy pierwszej.

        Zwykła warstwa konwolucyjna w trzeciej konfiguracji używa losowo zainicjalizowanego
        banku filtrów 1D. Końcowa warstwa klasyfikacyjna to regresja logistyczna trenowana
        na cechach statystycznych z map konwolucyjnych.
        """
    ),
    code(
        """
        def conv1d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            return correlate(x, kernel, mode="same")


        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(x, 0)


        def pool_stats(feature_maps: list[np.ndarray]) -> np.ndarray:
            features = []
            for fmap in feature_maps:
                fmap = np.asarray(fmap, dtype=np.float32)
                features.extend(
                    [
                        float(fmap.mean()),
                        float(fmap.std()),
                        float(fmap.max()),
                        float(fmap.min()),
                        float(np.percentile(fmap, 25)),
                        float(np.percentile(fmap, 50)),
                        float(np.percentile(fmap, 75)),
                        float(np.mean(np.abs(fmap))),
                    ]
                )
            return np.array(features, dtype=np.float32)


        class WaveletLayer1D:
            def __init__(self, wavelet: str = "db4"):
                w = pywt.Wavelet(wavelet)
                self.wavelet = wavelet
                self.filters = [
                    np.array(w.dec_lo[::-1], dtype=np.float32),
                    np.array(w.dec_hi[::-1], dtype=np.float32),
                ]

            def transform_one(self, signals: list[np.ndarray]) -> list[np.ndarray]:
                out = []
                for signal in signals:
                    for kernel in self.filters:
                        fmap = relu(conv1d_same(signal, kernel))[::2]
                        out.append(fmap.astype(np.float32))
                return out


        class RandomConvLayer1D:
            def __init__(self, n_filters: int = 8, kernel_size: int = 9, random_state: int = 42):
                rng = np.random.default_rng(random_state)
                scale = np.sqrt(2 / kernel_size)
                self.filters = rng.normal(0, scale, size=(n_filters, kernel_size)).astype(np.float32)
                self.filters -= self.filters.mean(axis=1, keepdims=True)

            def transform_one(self, signals: list[np.ndarray]) -> list[np.ndarray]:
                out = []
                for signal in signals:
                    for kernel in self.filters:
                        out.append(relu(conv1d_same(signal, kernel)).astype(np.float32))
                return out


        class WaveletConvECGNet:
            def __init__(
                self,
                architecture: str,
                wavelet: str = "db4",
                random_state: int = 42,
            ):
                self.architecture = architecture
                self.wavelet = wavelet
                self.random_state = random_state

            def _layers(self):
                layers = [WaveletLayer1D(self.wavelet)]
                if self.architecture == "2 warstwy falkowe":
                    layers.append(WaveletLayer1D(self.wavelet))
                elif self.architecture == "1 falkowa + 1 konwolucyjna":
                    layers.append(RandomConvLayer1D(n_filters=8, kernel_size=9, random_state=self.random_state))
                elif self.architecture != "1 warstwa falkowa":
                    raise ValueError(f"Nieznana architektura: {self.architecture}")
                return layers

            def transform(self, X: np.ndarray) -> np.ndarray:
                layers = self._layers()
                rows = []
                for frame in X:
                    maps = [frame.astype(np.float32)]
                    for layer in layers:
                        maps = layer.transform_one(maps)
                    rows.append(pool_stats(maps))
                return np.vstack(rows).astype(np.float32)

            def fit(self, X: np.ndarray, y: np.ndarray):
                features = self.transform(X)
                self.classifier_ = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=800,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                )
                self.classifier_.fit(features, y)
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                return self.classifier_.predict(self.transform(X))
        """
    ),
    code(
        """
        ARCHITECTURES = [
            "1 warstwa falkowa",
            "2 warstwy falkowe",
            "1 falkowa + 1 konwolucyjna",
        ]
        WAVELET = "db4"

        preview_n = min(3, len(X_frames))
        for arch in ARCHITECTURES:
            model = WaveletConvECGNet(architecture=arch, wavelet=WAVELET, random_state=RANDOM_STATE)
            features = model.transform(X_frames[:preview_n])
            print(f"{arch}: macierz cech dla {preview_n} ramek = {features.shape}")
        """
    ),
    md(
        """
        ## Walidacja krzyżowa po pacjentach

        Użyty jest `StratifiedGroupKFold`: etykiety są stratyfikowane, ale grupą jest
        rekord/pacjent. Dla każdego foldu sprawdzany jest warunek rozłączności grup
        treningowych i testowych. Raportowane metryki to accuracy, macro precision,
        macro recall, macro F1 i średnie bezwzględne MCC.
        """
    ),
    code(
        """
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


        def normalize_frames_by_train(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
            mean = X[train_idx].mean()
            std = X[train_idx].std() + 1e-8
            return (X[train_idx] - mean) / std, (X[test_idx] - mean) / std


        def evaluate_architecture(architecture: str):
            fold_rows = []
            confusion_sum = np.zeros((2, 2), dtype=int)
            for fold, (train_idx, test_idx) in enumerate(cv.split(X_frames, y, groups=groups), start=1):
                train_groups = set(groups[train_idx])
                test_groups = set(groups[test_idx])
                assert train_groups.isdisjoint(test_groups)

                X_train, X_test = normalize_frames_by_train(X_frames, train_idx, test_idx)
                clf = WaveletConvECGNet(
                    architecture=architecture,
                    wavelet=WAVELET,
                    random_state=RANDOM_STATE + fold,
                )
                clf.fit(X_train, y[train_idx])
                pred = clf.predict(X_test)

                row = {
                    "architecture": architecture,
                    "fold": fold,
                    "train_records": len(train_groups),
                    "test_records": len(test_groups),
                    "test_samples": len(test_idx),
                    "accuracy": accuracy_score(y[test_idx], pred),
                    "precision_macro": precision_score(y[test_idx], pred, average="macro", zero_division=0),
                    "recall_macro": recall_score(y[test_idx], pred, average="macro", zero_division=0),
                    "f1_macro": f1_score(y[test_idx], pred, average="macro", zero_division=0),
                    "mcc_abs": abs(matthews_corrcoef(y[test_idx], pred)),
                }
                fold_rows.append(row)
                confusion_sum += confusion_matrix(y[test_idx], pred, labels=[0, 1])
                print(
                    f"{architecture}, fold {fold}: "
                    f"acc={row['accuracy']:.3f}, f1={row['f1_macro']:.3f}, "
                    f"|MCC|={row['mcc_abs']:.3f}, test records={sorted(test_groups)}"
                )
            return pd.DataFrame(fold_rows), confusion_sum


        all_results = []
        confusion_by_architecture = {}
        for architecture in ARCHITECTURES:
            fold_df, cm = evaluate_architecture(architecture)
            all_results.append(fold_df)
            confusion_by_architecture[architecture] = cm

        results = pd.concat(all_results, ignore_index=True)
        display(results)
        """
    ),
    code(
        """
        metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc_abs"]
        summary = results.groupby("architecture")[metric_cols].agg(["mean", "std"]).round(4)
        display(summary)

        ranking = results.groupby("architecture")[metric_cols].mean().sort_values("f1_macro", ascending=False).round(4)
        display(ranking)
        print(f"Najlepsza konfiguracja wg średniego F1 macro: {ranking.index[0]}")
        """
    ),
    code(
        """
        plot_summary = results.groupby("architecture")[metric_cols].mean().loc[ARCHITECTURES]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        x = np.arange(len(metric_cols))
        width = 0.25
        for idx, architecture in enumerate(ARCHITECTURES):
            values = plot_summary.loc[architecture, metric_cols].to_numpy()
            axes[0].bar(x + (idx - 1) * width, values, width=width, label=architecture)
        axes[0].set_title("Średnie metryki z walidacji krzyżowej")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(["accuracy", "precision", "recall", "f1", "|MCC|"], rotation=20)
        axes[0].set_ylim(0, 1)
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.25)

        fold_metric = results.pivot(index="fold", columns="architecture", values="f1_macro")
        fold_metric[ARCHITECTURES].plot(ax=axes[1], marker="o")
        axes[1].set_title("F1 macro w kolejnych foldach")
        axes[1].set_xlabel("fold")
        axes[1].set_ylabel("F1 macro")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        plt.show()
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, len(ARCHITECTURES), figsize=(13, 3.6), constrained_layout=True)
        for ax, architecture in zip(axes, ARCHITECTURES):
            disp = ConfusionMatrixDisplay(
                confusion_by_architecture[architecture],
                display_labels=["bez arytmii", "arytmia"],
            )
            disp.plot(ax=ax, colorbar=False)
            ax.set_title(architecture)
        plt.show()
        """
    ),
    md(
        """
        ## Wnioski

        Notebook porównuje trzy konfiguracje sieci konwolucyjnych z warstwami falkowymi
        na ramkach 1D EKG. Walidacja jest grupowana po rekordzie/pacjencie, więc ten sam
        pacjent nie trafia jednocześnie do treningu i testu. Wyniki końcowe należy czytać
        z tabel `summary` i `ranking` powyżej: zawierają średnie oraz odchylenia standardowe
        dla accuracy, precision, recall, F1 oraz średniego bezwzględnego MCC.

        Najbardziej wymagającą metryką jest MCC, bo uwzględnia wszystkie pola macierzy
        pomyłek. W tym zadaniu raportowane jest `abs(MCC)` zgodnie z wymaganiem, a makro
        uśrednianie precision/recall/F1 ogranicza wpływ ewentualnej nierównowagi klas w
        poszczególnych foldach.
        """
    ),
]

nbf.write(nb, NOTEBOOK)
print(f"Notebook zapisany: {NOTEBOOK}")
