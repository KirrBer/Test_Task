"""
Подбор гиперпараметров CatBoost под целевую метрику (не AUC).

Objective:
  --objective f05                  F-мера с beta=0.5
  --objective recall_at_precision  max recall при precision >= --min-precision
  --objective tp_at_fp             max TP при FP <= --max-fp

Кеш фич в cache/features.pkl (пересчёт — через --rebuild-cache).

Запуск:
  python optuna_tune.py --data-dir .\\data --n-trials 30 --objective f05
  python optuna_tune.py --data-dir .\\data --n-trials 50 --objective recall_at_precision --min-precision 0.98
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from features import (
    FIELD_NAMES,
    extract_features,
    find_data_dir,
    load_subdivisions,
    passport_valid,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    raise SystemExit(1)


# ---------------------------------------------------------------- data

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


def find_subdivisions(data_dir):
    for cand in (
        Path(data_dir).parent / "reference" / "subdivisions.csv",
        Path(data_dir) / "reference" / "subdivisions.csv",
        Path(data_dir) / "subdivisions.csv",
    ):
        if cand.exists():
            return cand
    return None


def build_dataset(data_dir, cache_path, rebuild=False):
    if cache_path.exists() and not rebuild:
        print(f"[tune] loading features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[tune] building features from {data_dir}")
    preds = load_jsonl(data_dir / "train_predictions.jsonl")
    labels = {r["doc_id"]: r for r in load_jsonl(data_dir / "train_labels.jsonl")}
    subdiv = find_subdivisions(data_dir)
    subdivisions = load_subdivisions(subdiv) if subdiv else {}

    X_rows, y_rows = [], []
    for doc in preds:
        lab = labels.get(doc["doc_id"])
        if lab is None:
            continue
        lf = label_fields(lab)
        feats, corr = extract_features(doc, subdivisions=subdivisions)
        match = all(norm(corr.get(f, "")) == norm(lf.get(f, "")) for f in FIELD_NAMES)
        valid = passport_valid(lf)
        X_rows.append(feats)
        y_rows.append(1 if (match and valid) else 0)

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.array(y_rows, dtype=int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((X, y), f)
    print(f"[tune] cached {X.shape} -> {cache_path}")
    return X, y


# ---------------------------------------------------------------- metrics

def select_threshold_fp(oof, y, max_fp):
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


def select_threshold_precision(oof, y, min_precision):
    order = np.argsort(-oof)
    y_s = y[order]
    p_s = oof[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    total_pos = y.sum()
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)
    recall = tp / total_pos if total_pos else np.zeros_like(tp)
    valid = precision >= min_precision
    if not valid.any():
        thr, tp_f, fp_f = select_threshold_fp(oof, y, 0)
        return thr, tp_f, fp_f, 1.0, tp_f / total_pos if total_pos else 0.0
    i = np.where(valid)[0][-1]
    tp_f, fp_f = int(tp[i]), int(fp[i])
    prec = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 1.0
    rec = tp_f / total_pos if total_pos else 0.0
    return float(p_s[i]), tp_f, fp_f, prec, rec


def compute_metric(oof, y, objective, max_fp=0, min_precision=0.97):
    if objective == "f05":
        _, tp, fp = select_threshold_fp(oof, y, 0)
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / y.sum() if y.sum() else 0.0
        if p + r == 0:
            return 0.0
        return 1.25 * p * r / (0.25 * p + r)
    if objective == "recall_at_precision":
        _, tp, fp, p, r = select_threshold_precision(oof, y, min_precision)
        return r
    if objective == "tp_at_fp":
        _, tp, fp = select_threshold_fp(oof, y, max_fp)
        return float(tp)
    raise ValueError(objective)


# ---------------------------------------------------------------- optuna

def suggest_params(trial, base_loss):
    params = dict(
        iterations=trial.suggest_int("iterations", 80, 900, log=True),
        depth=trial.suggest_int("depth", 1, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.002, 0.15, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 40.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.005, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.5),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 150),
        border_count=trial.suggest_categorical("border_count", [16, 32, 64, 128, 254]),
        grow_policy=trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise"]),
        random_seed=40,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
        eval_metric="AUC",
        loss_function=base_loss,
    )
    if base_loss == "Logloss":
        # bal = trial.suggest_categorical("auto_class_weights", ["Balanced", "None"])
        params["auto_class_weights"] = "Balanced"
    return params


def make_objective(X, y, args, skf):
    def objective(trial):
        params = suggest_params(trial, args.loss)

        oof = np.zeros(len(y))
        fold_scores = []
        for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
            try:
                clf = CatBoostClassifier(**params)
            except Exception as e:
                raise optuna.TrialPruned() from e
            clf.fit(X.iloc[tr], y[tr])
            oof[va] = clf.predict_proba(X.iloc[va])[:, 1]

            # Промежуточная метрика по обработанным фолдам — для pruning
            score_now = compute_metric(
                oof[va], y[va], args.objective,
                max_fp=max(1, args.max_fp // 5),
                min_precision=args.min_precision,
            )
            fold_scores.append(score_now)
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = compute_metric(
            oof, y, args.objective,
            max_fp=args.max_fp,
            min_precision=args.min_precision,
        )
        return float(score)
    return objective


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None,
                        help="секунд на весь study")
    parser.add_argument("--objective", choices=("f05", "recall_at_precision", "tp_at_fp"),
                        default="recall_at_precision")
    parser.add_argument("--max-fp", type=int, default=0)
    parser.add_argument("--min-precision", type=float, default=0.98)
    parser.add_argument("--loss", default="Logloss",
                        help="Focal:...")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve() if args.models_dir \
        else Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir \
        else Path(__file__).resolve().parent / "cache"
    cache_path = cache_dir / "features.pkl"

    data_dir = find_data_dir(args.data_dir)
    if data_dir is None:
        print("[tune] ERROR: не нашёл данные")
        raise SystemExit(2)

    X, y = build_dataset(data_dir, cache_path, rebuild=args.rebuild_cache)
    print(f"[tune] X={X.shape}  pos={y.sum()} ({y.mean()*100:.2f}%)")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize",
                                sampler=sampler, pruner=pruner)

    print(f"[tune] objective={args.objective}  trials={args.n_trials}")
    study.optimize(
        make_objective(X, y, args, skf),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    print(f"\n[tune] best value ({args.objective}) = {study.best_value:.4f}")
    print("[tune] best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out = {
        "objective": args.objective,
        "best_value": study.best_value,
        "n_trials": args.n_trials,
        "best_params": study.best_params,
        "loss_function": args.loss,
        "max_fp": args.max_fp,
        "min_precision": args.min_precision,
        "n_folds": args.n_folds,
    }
    params_path = models_dir / "best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[tune] saved -> {params_path}")

    # Топ-5 trials для сравнения
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False).head(5)
    print("\n[tune] top-5 trials:")
    cols = ["number", "value"] + [c for c in df.columns
                                   if c.startswith("params_")]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()