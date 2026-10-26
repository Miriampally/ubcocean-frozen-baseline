#!/usr/bin/env python3
"""
Split diagnostics + slide-grouped re-run of the frozen-feature pipeline.

WHAT THIS DOES
  Step A  diagnose : checks whether your CURRENT train/test split leaks slides
                     and whether test counts are exactly 20% of every class.
  Step B  rerun    : re-runs the whole pipeline with a slide-grouped split:
                       - grouped 80/20 split (StratifiedGroupKFold, 1 fold = test)
                       - grouped 5-fold CV inside the training partition
                       - PCA(95% var) and SMOTE fitted on training data only
                       - ablations (a)-(d) for Table 2
                       - slide-cluster bootstrap 95% CIs
                       - slide-level aggregation (mean of patch probabilities)
                       - several split seeds (variability across partitions)
  Outputs CSV/JSON files you can copy into the yellow placeholders.

INPUTS YOU MUST PROVIDE
  --emb   .npy float array, shape (N, 1280): EfficientNetB0 embeddings, one row per patch
  --meta  .csv with one row per patch, same order as --emb, columns:
            label     class name or int (CC, EC, HGSC, LGSC, MC)
            slide_id  the slide/image id the patch came from (UBC-OCEAN image_id)
          optional:
            patient_id  use as the group instead of slide_id if you have it
            is_test     0/1 flag marking your CURRENT test split (used by `diagnose`)

NOTE: as far as I know UBC-OCEAN's train.csv has no patient ID, so grouping is
by slide (image_id). Say so in the paper: several slides may come from one patient.

USAGE
  # EASIEST (uses your features.npz directly, reproduces your old split):
  python grouped_rerun.py diagnose --npz features.npz
  python grouped_rerun.py rerun --npz features.npz --out results
  # or with separate files:
  python grouped_rerun.py diagnose --meta meta.csv
  python grouped_rerun.py rerun --emb emb.npy --meta meta.csv --out results --seeds 0 1 2 3 4
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- helpers
def load(meta_path, emb_path=None):
    meta = pd.read_csv(meta_path)
    meta["group"] = meta["patient_id"] if "patient_id" in meta.columns else meta["slide_id"]
    classes = sorted(meta["label"].unique())
    meta["y"] = meta["label"].map({c: i for i, c in enumerate(classes)})
    emb = np.load(emb_path) if emb_path else None
    if emb is not None:
        assert len(emb) == len(meta), "embeddings and metadata must have the same length/order"
    return meta, emb, classes


def load_npz(npz_path):
    """Read features.npz from 02_feature_extraction_agent.py (X, y, label_names, patch_paths).
    Slide id = number before '_tile' in the patch filename (e.g. 44962_tile_0003.npy -> 44962).
    Also reproduces the ORIGINAL split of 03_train_evaluate_agent.py
    (train_test_split(test_size=0.2, stratify=y, random_state=42)) as `is_test`."""
    import re
    from sklearn.model_selection import train_test_split
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32); y = z["y"].astype(int)
    names = [str(c) for c in z["label_names"]]; paths = [str(p) for p in z["patch_paths"]]
    ids = []
    for p in paths:
        m = re.search(r"(\d+)_tile", p.replace("\\", "/").split("/")[-1])
        ids.append(m.group(1) if m else None)
    meta = pd.DataFrame({"label": [names[i] for i in y], "slide_id": ids, "path": paths})
    bad = meta.slide_id.isna().sum()
    if bad:
        raise SystemExit(f"{bad} patch names have no '<id>_tile' pattern; adjust the regex in load_npz().")
    idx = np.arange(len(y))
    _, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    meta["is_test"] = 0; meta.loc[te, "is_test"] = 1
    meta["group"] = meta["slide_id"]; meta["y"] = y
    return meta, X, names


def diagnose(meta_path=None, npz_path=None):
    if npz_path:
        meta, _, classes = load_npz(npz_path)
    else:
        meta, _, classes = load(meta_path)
    print("=== Class totals and (if is_test given) test fractions ===")
    tot = meta.groupby("label").size()
    out = pd.DataFrame({"total": tot})
    if "is_test" in meta.columns:
        te = meta[meta.is_test == 1].groupby("label").size()
        out["test"] = te
        out["test_fraction"] = (te / tot).round(4)
        print(out)
        fr = out["test_fraction"].values
        print("\nMax deviation of any class from 0.2000:", float(np.abs(fr - 0.2).max()))
        print("-> If every class is ~0.2000 the split was almost certainly a stratified "
              "PATCH-level split, not a grouped one.")
        tr_g = set(meta[meta.is_test == 0].group)
        te_g = set(meta[meta.is_test == 1].group)
        both = tr_g & te_g
        print(f"\nGroups (slides/patients) in train: {len(tr_g)}, in test: {len(te_g)}")
        print(f"Groups appearing in BOTH train and test: {len(both)}")
        if both:
            n_leak = meta[(meta.is_test == 1) & meta.group.isin(both)].shape[0]
            print(f"Test patches whose slide is also in train: {n_leak} "
                  f"({100 * n_leak / (meta.is_test == 1).sum():.1f}% of test)")
            print("-> LEAKAGE CONFIRMED. Re-run with `rerun` and rewrite Section 3.4.")
        else:
            print("-> No slide overlap: split is grouped. Report slide/patient counts in Section 3.4 "
                  "and explain why class fractions are ~20%.")
    else:
        print(out)
        print("Add an `is_test` column (0/1) to check your existing split.")
    print("\n=== Patches per group (sampling uniformity) ===")
    print(meta.groupby("group").size().describe().round(1))
    print("\n=== Groups per class ===")
    print(meta.groupby("label").group.nunique())


def build_clf(seed, n_cls):
    return lgb.LGBMClassifier(objective="multiclass", num_class=n_cls, n_estimators=300,
                              learning_rate=0.05, num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    # Replace with your tuned hyperparameters before the final run.


def make_features(Xtr, Xte, use_pca, seed):
    if not use_pca:
        return Xtr, Xte
    p = PCA(n_components=0.95, svd_solver="full", random_state=seed).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)


def fit_predict(cfg, Xtr, ytr, Xte, seed, n_cls):
    """cfg: dict(model='lgbm'|'lr', pca=bool, balance='smote'|'weights'|None)"""
    A, B = make_features(Xtr, Xte, cfg["pca"], seed)
    if cfg["model"] == "lr":
        sc = StandardScaler().fit(A)
        m = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=seed)
        m.fit(sc.transform(A), ytr)
        return m.predict_proba(sc.transform(B))
    clf = build_clf(seed, n_cls)
    if cfg["balance"] == "smote":
        A, ytr = SMOTE(random_state=seed).fit_resample(A, ytr)
        clf.fit(A, ytr)
    elif cfg["balance"] == "weights":
        clf.set_params(class_weight="balanced"); clf.fit(A, ytr)
    else:
        clf.fit(A, ytr)
    return clf.predict_proba(B)


def metrics(y, P):
    pred = P.argmax(1)
    return dict(accuracy=accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro"),
                weighted_f1=f1_score(y, pred, average="weighted"),
                balanced_acc=balanced_accuracy_score(y, pred),
                macro_auc=roc_auc_score(y, P, multi_class="ovr", average="macro"))


def cluster_bootstrap(y, P, groups, B=1000, seed=0):
    """Resample whole slides with replacement; recompute metrics on the pooled patches."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups); ug = np.unique(g)
    idx_by = {k: np.where(g == k)[0] for k in ug}
    rows = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[ii])) < P.shape[1]:
            continue
        rows.append(metrics(y[ii], P[ii]))
    d = pd.DataFrame(rows)
    return {c: (float(d[c].quantile(.025)), float(d[c].quantile(.975))) for c in d.columns}


def slide_level(y, P, groups):
    df = pd.DataFrame(P); df["g"] = groups; df["y"] = y
    m = df.groupby("g").agg({**{c: "mean" for c in range(P.shape[1])}, "y": "first"})
    Ps = m[list(range(P.shape[1]))].values; ys = m["y"].values.astype(int)
    out = dict(n_slides=int(len(ys)), balanced_acc=float(balanced_accuracy_score(ys, Ps.argmax(1))),
               accuracy=float(accuracy_score(ys, Ps.argmax(1))))
    return out, ys, Ps, m.index.values


# ---------------------------------------------------------------- main experiment
FULL = dict(model="lgbm", pca=True, balance="smote")
ABLATIONS = {
    "(a) LogReg raw 1280-d, class weights": dict(model="lr", pca=False, balance=None),
    "(b) LightGBM raw embeddings, no PCA": dict(model="lgbm", pca=False, balance="weights"),
    "(c) PCA + LightGBM, class weights (no SMOTE)": dict(model="lgbm", pca=True, balance="weights"),
    "(d) Full: PCA + SMOTE + LightGBM": FULL,
}


def grouped_split(meta, seed):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(meta, meta.y, meta.group))
    assert not (set(meta.group.iloc[tr]) & set(meta.group.iloc[te]))
    return tr, te


def grouped_cv_score(X, y, groups, seed, n_cls):
    """Grouped 5-fold CV on the TRAINING partition; PCA + SMOTE inside each fold."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s = []
    for a, b in sgkf.split(X, y, groups):
        P = fit_predict(FULL, X[a], y[a], X[b], seed, n_cls)
        f1s.append(f1_score(y[b], P.argmax(1), average="macro"))
    return float(np.mean(f1s)), float(np.std(f1s))


def rerun(emb_path, meta_path, out, seeds, B, npz_path=None):
    os.makedirs(out, exist_ok=True)
    if npz_path:
        meta, X, classes = load_npz(npz_path)
    else:
        meta, X, classes = load(meta_path, emb_path)
    y = meta.y.values; g = meta.group.values; n_cls = len(classes)
    summary, per_seed = [], []
    for s in seeds:
        tr, te = grouped_split(meta, s)
        rec = dict(seed=s, train_patches=len(tr), test_patches=len(te),
                   train_groups=int(meta.group.iloc[tr].nunique()),
                   test_groups=int(meta.group.iloc[te].nunique()))
        cvm, cvs = grouped_cv_score(X[tr], y[tr], g[tr], s, n_cls)
        rec["grouped_cv_macro_f1_mean"], rec["grouped_cv_macro_f1_sd"] = cvm, cvs
        for name, cfg in ABLATIONS.items():
            P = fit_predict(cfg, X[tr], y[tr], X[te], s, n_cls)
            m = metrics(y[te], P)
            for k, v in m.items():
                rec[f"{name} | {k}"] = v
            if cfg is FULL:
                ci = cluster_bootstrap(y[te], P, g[te], B=B, seed=s)
                sl, ys, Ps, ids = slide_level(y[te], P, g[te])
                rec["full_slide_level"] = sl
                rec["full_ci"] = ci
                pd.DataFrame(P, columns=classes).assign(
                    y=y[te], group=g[te]).to_csv(f"{out}/test_probs_seed{s}.csv", index=False)
        per_seed.append(rec)
        print(f"seed {s}: grouped CV macro-F1 {cvm:.3f}±{cvs:.3f} | "
              f"test macro-F1 {rec['(d) Full: PCA + SMOTE + LightGBM | macro_f1']:.3f} | "
              f"slide-level bal.acc {rec['full_slide_level']['balanced_acc']:.3f}")
    json.dump(per_seed, open(f"{out}/per_seed.json", "w"), indent=2, default=float)

    # Table 2 (mean over seeds; CIs shown for the first seed's test set)
    rows = []
    for name in ABLATIONS:
        r = {"Configuration": name}
        for k in ["macro_f1", "balanced_acc", "macro_auc"]:
            v = [d[f"{name} | {k}"] for d in per_seed]
            r[k + "_mean"] = np.mean(v); r[k + "_sd_across_splits"] = np.std(v)
        rows.append(r)
    t2 = pd.DataFrame(rows).round(3); t2.to_csv(f"{out}/table2_ablation.csv", index=False)
    print("\nTable 2 (mean over grouped splits):\n", t2.to_string(index=False))

    gap = pd.DataFrame([{ "seed": d["seed"],
                          "grouped_cv_macro_f1": d["grouped_cv_macro_f1_mean"],
                          "test_macro_f1": d["(d) Full: PCA + SMOTE + LightGBM | macro_f1"]} for d in per_seed])
    gap["gap"] = gap.grouped_cv_macro_f1 - gap.test_macro_f1
    gap.round(3).to_csv(f"{out}/cv_test_gap.csv", index=False)
    print("\nGrouped CV vs test gap:\n", gap.round(3).to_string(index=False))
    print(f"\nFiles written to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose"); d.add_argument("--meta"); d.add_argument("--npz")
    r = sub.add_parser("rerun")
    r.add_argument("--emb"); r.add_argument("--meta"); r.add_argument("--npz")
    r.add_argument("--out", default="results"); r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose(a.meta, a.npz)
    else:
        rerun(a.emb, a.meta, a.out, a.seeds, a.boot, a.npz)
