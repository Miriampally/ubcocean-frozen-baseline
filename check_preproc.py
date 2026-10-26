"""
check_preproc.py -- which preprocessing options produced your preprocessed .npy files?

It replays the FIRST rows of preprocessing_manifest.csv through 01_preprocessing_agent.py with every
combination of normalisation (minmax/zscore) and denoising (gaussian/median/none), with augmentation on
and off, and compares the result with the stored .npy arrays. Because the agent seeds its random
generator (seed 42) at start, the first processed files can be reproduced exactly.
A max abs difference near 0 (< 1e-4) means that configuration is what you ran.

Usage (run with the Python that has opencv-python, numpy, pillow):
  python check_preproc.py --script "C:\\Users\\miria\\Downloads\\files123\\01_preprocessing_agent.py" ^
      --manifest "D:\\...\\preprocessed\\preprocessing_manifest.csv" ^
      --preprocessed_dir "D:\\...\\preprocessed" --n 4
"""
import argparse, csv, importlib.util, itertools, os, sys
from pathlib import Path
import numpy as np

def load_module(path):
    spec = importlib.util.spec_from_file_location("prep01", path)
    m = importlib.util.module_from_spec(spec); sys.modules["prep01"] = m; spec.loader.exec_module(m); return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--preprocessed_dir", required=True); ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    m = load_module(a.script)
    rows = []
    with open(a.manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r["source_path"])
            if len(rows) == a.n: break
    stored = []
    for sp in rows:
        p = Path(sp); cand = [Path(a.preprocessed_dir) / p.parent.name / (p.stem + ".npy"), Path(a.preprocessed_dir) / (p.stem + ".npy")]
        f = next((c for c in cand if c.exists()), None)
        if f is None: raise SystemExit(f"stored array not found for {sp}")
        stored.append(np.load(f))
    print("first manifest rows:", [Path(r).name for r in rows])
    results = []
    for aug in (True, False):
        for norm, den in itertools.product(["minmax", "zscore"], ["gaussian", "median", "none"]):
            cfg = m.PreprocessConfig(normalization=norm, denoise_method=den, augment=aug)
            agent = m.MultimodalPreprocessingAgent(cfg)
            diffs = []
            for sp, ref in zip(rows, stored):
                out = agent.process_one(Path(sp), modality_hint="histopathology").array
                diffs.append(float(np.abs(out - ref).max()) if out.shape == ref.shape else float("inf"))
            results.append((max(diffs), aug, norm, den, diffs))
    results.sort(key=lambda r: r[0])
    print("\nbest matches (max abs diff over the first %d patches):" % len(rows))
    for d, aug, norm, den, diffs in results[:6]:
        print(f"  augment={aug!s:5} normalization={norm:6} denoise={den:8} max diff {d:.6f}   per patch {[round(x,6) for x in diffs]}")
    exact = [r for r in results if r[0] < 1e-4]
    if exact:
        print("\nEXACT REPRODUCTION. Matching configuration(s):")
        for d, aug, norm, den, _ in exact: print(f"   augment={aug}, normalization={norm}, denoise={den}")
        if len({(r[1], r[3]) for r in exact}) == 1 and len({r[2] for r in exact}) > 1:
            print("   (minmax and zscore give identical arrays here, so the normalisation option cannot be told apart; the denoising and augmentation options are identified.)")
    else:
        print("\nNO exact reproduction. The files were probably made by an earlier version of the script, in a different "
              "processing order, with a different random seed, or with other settings; this test cannot confirm the options.")

if __name__ == "__main__":
    main()
