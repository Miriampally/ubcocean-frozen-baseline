#!/usr/bin/env python3
"""
slide_level.py  --  slide-level classification from frozen patch embeddings.

Each slide (UBC-OCEAN image_id) becomes ONE sample: its patch embeddings are pooled
(mean, or mean+max) into a single vector, and a classifier predicts the slide's subtype.
Every slide appears in exactly one test fold, so there is no patch-level leakage by design.
Evaluation = repeated stratified 5-fold CV over slides (out-of-fold predictions),
with 95% bootstrap CIs resampling slides.

Run AFTER grouped_rerun.py has finished (both use all CPU cores).

  python slide_level.py --npz features.npz --out results_slide --seeds 0 1 2 3 4

Needs: numpy pandas scikit-learn lightgbm  (same as grouped_rerun.py)
Reads grouped_rerun.py from the same folder (for load_npz).
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import lightgbm as lgb
from grouped_rerun import load_npz

warnings.filterwarnings("ignore")


def pool_slides(meta, X, mode):
    """Return slide-level matrix, labels, slide ids, n_patches."""
    ids, feats, ys, ns = [], [], [], []
    for sid, idx in meta.groupby("slide_id").indices.items():
        e = X[idx]
        v = e.mean(0) if mode == "mean" else np.concatenate([e.mean(0), e.max(0)])
        ids.append(sid); feats.append(v); ys.append(int(meta.y.values[idx][0])); ns.append(len(idx))
        assert len(set(meta.y.values[idx])) == 1, f"slide {sid} has mixed labels"
    return np.vstack(feats).astype(np.float32), np.array(ys), np.array(ids), np.array(ns)


def make_model(name, seed):
    if name == "logreg":
        return "logreg"
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=200, learning_rate=0.05,
                              num_leaves=15, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                              class_weight="balanced", min_child_samples=5,
                              random_state=seed, n_jobs=-1, verbose=-1)


def fit_predict(name, Xtr, ytr, Xte, seed, n_cls):
    if name == "logreg":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegressionCV(Cs=np.logspace(-4, -1, 6), cv=3, class_weight="balanced",
                                 max_iter=3000, scoring="balanced_accuracy", n_jobs=-1, random_state=seed)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))
    m = make_model(name, seed); m.fit(Xtr, ytr)
    return m.predict_proba(Xte)


def metrics(y, P):
    pred = P.argmax(1)
    return dict(balanced_acc=balanced_accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                accuracy=accuracy_score(y, pred),
                macro_auc=roc_auc_score(y, P, multi_class="ovr", average="macro"))


def boot_ci(y, P, B, seed):
    rng = np.random.default_rng(seed); n = len(y); rows = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < P.shape[1]:
            continue
        rows.append(metrics(y[i], P[i]))
    d = pd.DataFrame(rows)
    return {c: [float(d[c].quantile(.025)), float(d[c].quantile(.975))] for c in d.columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out", default="results_slide")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    meta, X, classes = load_npz(a.npz); n_cls = len(classes)
    summary, rows = {}, []
    for pool in ["mean", "meanmax"]:
        S, y, ids, ns = pool_slides(meta, X, pool)
        print(f"[{pool}] {len(y)} slides, feature dim {S.shape[1]}, class counts "
              f"{dict(zip(classes, np.bincount(y)))}")
        for name in ["logreg", "lgbm"]:
            per_seed, oof_all = [], []
            for s in a.seeds:
                skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=s)
                oof = np.zeros((len(y), n_cls))
                for tr, te in skf.split(S, y):
                    oof[te] = fit_predict(name, S[tr], y[tr], S[te], s, n_cls)
                per_seed.append(metrics(y, oof)); oof_all.append(oof)
                print(f"  {pool}/{name} seed {s}: bal.acc {per_seed[-1]['balanced_acc']:.3f}  "
                      f"macro-F1 {per_seed[-1]['macro_f1']:.3f}  AUC {per_seed[-1]['macro_auc']:.3f}", flush=True)
            Pm = np.mean(oof_all, axis=0)          # average OOF probs over repeats (for CI + confusion)
            ci = boot_ci(y, Pm, a.boot, 0)
            d = pd.DataFrame(per_seed)
            rec = dict(pooling=pool, model=name, n_slides=int(len(y)))
            for c in d.columns:
                rec[c + "_mean"] = float(d[c].mean()); rec[c + "_sd_across_repeats"] = float(d[c].std(ddof=0))
                rec[c + "_ci_lo"], rec[c + "_ci_hi"] = ci[c]
            rows.append(rec)
            cm = confusion_matrix(y, Pm.argmax(1))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"{a.out}/confusion_{pool}_{name}.csv")
            pd.DataFrame(Pm, columns=classes).assign(slide_id=ids, y=y, n_patches=ns).to_csv(
                f"{a.out}/oof_probs_{pool}_{name}.csv", index=False)
    t = pd.DataFrame(rows).round(3); t.to_csv(f"{a.out}/slide_level_results.csv", index=False)
    show = t[["pooling", "model", "n_slides", "balanced_acc_mean", "balanced_acc_ci_lo", "balanced_acc_ci_hi",
              "macro_f1_mean", "macro_auc_mean"]]
    print("\n", show.to_string(index=False)); print(f"\nWrote {a.out}/")


if __name__ == "__main__":
    main()
