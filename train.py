"""
Обучение accept/reject на CatBoost.

  y = 1  iff  corrected_fields == label_fields для всех 10 полей
              И passport_valid(label_fields) = True.

Порог выбирается под --max-fp (default 0): максимум TP при FP <= max_fp.
Дополнительно печатается sweep по precision для сравнения.

Запуск:
  python train.py --data-dir .\\data                       # FP=0
  python train.py --data-dir .\\data --max-fp 5            # практично
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import (
    FIELD_NAMES,
    describe_dir,
    extract_features,
    find_data_dir,
    load_subdivisions,
    passport_valid,
)


# ---------------------------------------------------------------- utils

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def label_fields(lab):
    if isinstance(lab.get("fields"), dict):
        return lab["fields"]
    return {k: v for k, v in lab.items() if k != "doc_id"}


def norm(v):
    s = "" if v is None else str(v)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- thresholds

def select_threshold(oof, y, max_fp=0):
    """Максимум TP при FP <= max_fp. Тай-брейк: positives первыми."""
    order = np.lexsort((-y, -oof))
    best_thr, best_tp, best_fp = 1.0, 0, 0
    tp = fp = 0
    for idx in order:
        if y[idx] == 1:
            tp += 1
        else:
            fp += 1
            if fp > max_fp:
                break
        if tp > best_tp:
            best_thr, best_tp, best_fp = float(oof[idx]), tp, fp
    return best_thr, best_tp, best_fp


def select_threshold_by_precision(oof, y, min_precision=0.98):
    """Максимум TP при precision >= min_precision."""
    order = np.argsort(-oof)
    y_sorted = y[order]
    p_sorted = oof[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = y.sum()
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)
    recall = tp / total_pos if total_pos else np.zeros_like(tp)
    valid = precision >= min_precision
    if not valid.any():
        thr, tp_f, fp_f = select_threshold(oof, y, max_fp=0)
        return thr, tp_f, fp_f, 1.0, tp_f / total_pos if total_pos else 0.0
    last_valid = np.where(valid)[0][-1]
    thr = float(p_sorted[last_valid])
    tp_f = int(tp[last_valid])
    fp_f = int(fp[last_valid])
    prec = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 1.0
    rec = tp_f / total_pos if total_pos else 0.0
    return thr, tp_f, fp_f, prec, rec


def sweep_fp(oof, y, budgets=(0, 1, 2, 3, 5, 10, 20, 50, 100, 200)):
    n = len(y)
    total_pos = y.sum()
    print("\n[train] FP-budget sweep (OOF):")
    print(f"  {'max_fp':>8} {'thr':>8} {'TP':>5} {'FP':>4} "
          f"{'P':>7} {'R':>7} {'accept':>7}")
    for b in budgets:
        thr, tp, fp = select_threshold(oof, y, max_fp=b)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / total_pos if total_pos else 0.0
        acc = (tp + fp) / n * 100
        print(f"  {b:>8} {thr:>8.4f} {tp:>5} {fp:>4} "
              f"{prec:>7.3f} {rec*100:>6.1f}% {acc:>6.1f}%")


def sweep_precision(oof, y,
                    targets=(0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999)):
    n = len(y)
    print("\n[train] precision → recall sweep (OOF):")
    print(f"  {'target_P':>8} {'thr':>8} {'TP':>5} {'FP':>4} "
          f"{'P_act':>7} {'R_act':>7} {'accept':>7}")
    for target in targets:
        thr, tp, fp, prec, rec = select_threshold_by_precision(oof, y, target)
        acc = (tp + fp) / n * 100
        print(f"  {target:>8.3f} {thr:>8.4f} {tp:>5} {fp:>4} "
              f"{prec:>7.3f} {rec*100:>6.1f}% {acc:>6.1f}%")


# ---------------------------------------------------------------- paths

def find_subdivisions(data_dir):
    for cand in (
        Path(data_dir).parent / "reference" / "subdivisions.csv",
        Path(data_dir) / "reference" / "subdivisions.csv",
        Path(data_dir) / "subdivisions.csv",
    ):
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------- model

def make_catboost(params=None):
    base = dict(
        iterations=688,
        depth=2,
        learning_rate=0.06822,
        l2_leaf_reg=1.9133,
        random_strength=0.1525,
        bagging_temperature=1.3388,
        min_data_in_leaf=133,
        border_count=64,
        grow_policy="SymmetricTree",
        loss_function='Logloss',
        eval_metric="AUC",
        random_seed=40,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    if params:
        base.update(params)
    return CatBoostClassifier(**base)

def load_best_params(models_dir):
    p = Path(models_dir) / "best_params.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        params = d.get("best_params") or {}
        loss_f = d.get('loss_function') or "Logloss"
        params["loss_function"] = loss_f
        print(f"[train] loaded hyperparams from {p} ({len(params)} keys)")
        return params
    except Exception as e:
        print(f"[train] WARN: не смог прочитать {p}: {e}")
        return {}
# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--min_precision", type=int, default=0.99,
                        help="min precision на OOF при выборе порога")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve() if args.models_dir \
        else Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir(args.data_dir)
    if data_dir is None:
        print("[train] ERROR: не нашёл *_predictions.jsonl")
        describe_dir(Path.cwd())
        describe_dir(Path(__file__).resolve().parent)
        raise SystemExit(2)

    preds_path = data_dir / "train_predictions.jsonl"
    labels_path = data_dir / "train_labels.jsonl"
    for p in (preds_path, labels_path):
        if not p.exists():
            print(f"[train] ERROR: {p} не найден.")
            describe_dir(data_dir)
            raise SystemExit(2)

    preds = load_jsonl(preds_path)
    labels = {r["doc_id"]: r for r in load_jsonl(labels_path)}
    print(f"[train] DATA={data_dir}  preds={len(preds)}  labels={len(labels)}")

    subdiv_path = find_subdivisions(data_dir)
    subdivisions = load_subdivisions(subdiv_path) if subdiv_path else {}
    print(f"[train] subdivisions: {len(subdivisions)} codes")

    # -------- dataset --------
    X_rows, y_rows, used_docs = [], [], []
    n_match = n_valid = 0
    n_mrz = n_mrz_parsed = n_mrz_all_ok = 0
    mrz_checks_total = mrz_checks_ok = 0

    for doc in preds:
        lab = labels.get(doc["doc_id"])
        if lab is None:
            continue
        lf = label_fields(lab)
        feats, corr = extract_features(doc, subdivisions=subdivisions)

        if feats.get("mrz_present"):
            n_mrz += 1
            if feats.get("mrz_parsed"):
                n_mrz_parsed += 1
                mrz_checks_total += int(feats["mrz_n_checks"])
                mrz_checks_ok += int(feats["mrz_n_checks_ok"])
                if feats["mrz_all_checks_ok"]:
                    n_mrz_all_ok += 1

        match = all(norm(corr.get(f, "")) == norm(lf.get(f, "")) for f in FIELD_NAMES)
        valid = passport_valid(lf)
        y = 1 if (match and valid) else 0

        n_match += int(match)
        n_valid += int(valid)
        X_rows.append(feats)
        y_rows.append(y)
        used_docs.append(doc)

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.array(y_rows, dtype=int)
    print(f"[train] X={X.shape}  positives={y.sum()} ({y.mean()*100:.2f}%)")
    print(f"[train] exact-match={n_match}  label-valid={n_valid}")
    print(f"[train] MRZ: present={n_mrz}  parsed={n_mrz_parsed}  "
          f"all_checks_ok={n_mrz_all_ok}")
    if mrz_checks_total:
        print(f"[train] MRZ checks: {mrz_checks_ok}/{mrz_checks_total} "
              f"({mrz_checks_ok/mrz_checks_total*100:.1f}%) passed")

    # -------- CV --------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    oof = np.zeros(len(y))

    params = load_best_params(models_dir)
    print(f"[train] 5-fold CV (CatBoost)...")
    for i, (tr, va) in enumerate(skf.split(X, y)):
        clf = make_catboost(params)
        clf.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]),
                use_best_model=False)
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
        fold_auc = roc_auc_score(y[va], oof[va]) if y[va].min() != y[va].max() else float("nan")
        print(f"  fold {i}: auc={fold_auc:.4f}")

    print(f"[train] OOF AUC={roc_auc_score(y, oof):.4f}")
    print(f"[train] OOF AP ={average_precision_score(y, oof):.4f}")

    # -------- sweeps --------
    sweep_fp(oof, y)
    sweep_precision(oof, y)

    # -------- final threshold (по max_fp) --------
    thr, tp, fp, prec, rec = select_threshold_by_precision(oof, y, args.min_precision)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / y.sum() if y.sum() else 0.0
    print(f"\n[train] CHOSEN (min_precision={args.min_precision}):")
    print(f"[train]   thr={thr:.4f}")
    print(f"[train]   TP={tp}  FP={fp}  FN={y.sum()-tp}")
    print(f"[train]   precision={prec:.4f}  recall={rec:.4f}")
    print(f"[train]   accept_rate={(tp+fp)/len(y)*100:.2f}%")

    # -------- diagnostics: топ negatives --------
    neg_idx = np.where(y == 0)[0]
    if len(neg_idx):
        top = neg_idx[np.argsort(-oof[neg_idx])[:10]]
        diag_cols = [
            "mrz_present", "mrz_all_checks_ok",
            "mrz_surname_charsim", "name_sim_mean",
            "mrz_number_full_eq", "sharpness",
            "doc_mean_cc", "subdiv_code_in_ref", "passport_valid",
        ]
        diag_cols = [c for c in diag_cols if c in X.columns]
        print("\n[train] top-10 negatives by prob:")
        for rank, i in enumerate(top, 1):
            row = X.iloc[i][diag_cols].to_dict()
            info = "  ".join(f"{k}={row[k]:.2f}" for k in diag_cols)
            print(f"  {rank:>2}. {used_docs[i]['doc_id']:<12} "
                  f"prob={oof[i]:.4f}  {info}")

    # -------- retrain on full --------
    print("[train] retrain on full data...")
    final = make_catboost(params)
    final.fit(X, y)

    imp = pd.Series(final.get_feature_importance(), index=X.columns) \
            .sort_values(ascending=False)
    print("[train] top-50 features:")
    print(imp.head(50).to_string())

    # -------- save --------
    model_path = models_dir / "accept_model.cbm"
    final.save_model(str(model_path))
    joblib.dump(
        {
            "model_type": "catboost",
            "model_path": str(model_path),
            "threshold": thr,
            "expected_precision": prec,
            "expected_recall": rec,
            "field_names": FIELD_NAMES,
            "feature_names": list(X.columns),
        },
        models_dir / "accept_model.pkl",
    )
    print(f"[train] saved -> {model_path}")
    print(f"[train] saved -> {models_dir / 'accept_model.pkl'}")


if __name__ == "__main__":
    main()