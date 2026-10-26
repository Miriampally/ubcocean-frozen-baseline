import argparse, os, re
import numpy as np

EXPECTED = {"CC": 1434, "EC": 1587, "HGSC": 2988, "LGSC": 590, "MC": 594}

def cmd_filter(a):
    from PIL import Image
    got = {}
    for cls in sorted(d for d in os.listdir(a.patches) if os.path.isdir(os.path.join(a.patches, d))):
        flagged = total = 0
        for f in sorted(os.listdir(os.path.join(a.patches, cls))):
            if os.path.splitext(f)[1].lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            arr = np.array(Image.open(os.path.join(a.patches, cls, f)).convert("RGB"))
            black = np.all(arr < 15, axis=-1).mean()
            white = np.all(arr > 240, axis=-1).mean()
            total += 1
            flagged += int(black > 0.05 or white > 0.85)
        got[cls] = (total, flagged)
        print(cls, "tiles", total, "flagged", flagged, flush=True)
    print("total flagged:", sum(v[1] for v in got.values()))
    ok = all(got.get(k, (0, -1))[1] == v for k, v in EXPECTED.items())
    print("EXACT MATCH: default thresholds were used." if ok else
          "NO exact match. Expected flagged counts: " + str(EXPECTED))

def cmd_tma(a):
    import pandas as pd
    t = pd.read_csv(a.train_csv)
    t["image_id"] = t["image_id"].astype(str)
    z = np.load(a.npz, allow_pickle=True)
    ids = {re.search(r"(\d+)_tile", str(p)).group(1) for p in z["patch_paths"]}
    print("train.csv rows:", len(t), "is_tma:", t["is_tma"].value_counts().to_dict())
    u = t[t.image_id.isin(ids)]
    print("slides in features.npz:", len(u), "is_tma:", u["is_tma"].value_counts().to_dict())
    print("labels in train.csv:", t["label"].value_counts().to_dict())

def cmd_features(a):
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    z = np.load(a.npz, allow_pickle=True)
    X = z["X"]
    paths = [str(p) for p in z["patch_paths"]]
    idx = np.linspace(0, len(paths) - 1, a.n).astype(int)
    imgs = np.stack([np.load(paths[i]) for i in idx]).astype(np.float32)
    m = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    E = m.predict(preprocess_input(imgs * 255.0), verbose=0)
    d = np.abs(E - X[idx])
    rel = d.max() / (np.abs(X[idx]).max() + 1e-12)
    print("TensorFlow", tf.__version__, "max abs diff", float(d.max()), "relative", float(rel))
    print("REPRODUCES features.npz: cite this TensorFlow version." if rel < 1e-3 else
          "DOES NOT match: a different environment or preprocessing made features.npz.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    f = sp.add_parser("filter"); f.add_argument("--patches", required=True)
    t = sp.add_parser("tma"); t.add_argument("--train_csv", required=True); t.add_argument("--npz", required=True)
    g = sp.add_parser("features"); g.add_argument("--npz", required=True); g.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    {"filter": cmd_filter, "tma": cmd_tma, "features": cmd_features}[a.c](a)
