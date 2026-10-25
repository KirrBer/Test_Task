#!/usr/bin/env python3
"""
postprocess.py — accept/reject для паспортов РФ.

Три обученных режима под три точки на PR-кривой:
  mode 98   high-recall    (P>=0.98)
  mode 99   balanced       (P>=0.99)
  mode 100  high-precision (P=1.00)

Каждый режим = models/mode_<N>/accept_model.pkl + accept_model.cbm.

Примеры:
  python postprocess.py data/test_predictions.jsonl answers.jsonl --mode 99
  python postprocess.py data/test_predictions.jsonl answers/ --all
  python postprocess.py data/train_predictions.jsonl /tmp/ans/ --all \
      --labels data/train_labels.jsonl --compare

Если файлы моделей отсутствуют — скрипт печатает причину и выходит с кодом 2.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import (
    FIELD_NAMES,
    extract_features,
    find_data_dir,
    load_subdivisions,
    passport_valid,
)


# ---------------------------------------------------------------- constants

MODES = ("98", "99", "100")
MODE_DESCR = {
    "98":  "high-recall    (P>=0.98)",
    "99":  "balanced       (P>=0.99)",
    "100": "high-precision (P=1.00)",
}


# ---------------------------------------------------------------- io helpers

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


def find_subdiv_path(data_dir: Path):
    for cand in (
        data_dir.parent / "reference" / "subdivisions.csv",
        data_dir / "reference" / "subdivisions.csv",
        data_dir / "subdivisions.csv",
    ):
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------- model loading

def load_bundle(models_root: Path, mode: str):
    mode_dir = models_root / f"mode_{mode}"
    bundle_path = mode_dir / "accept_model.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"нет бандла: {bundle_path}")

    bundle = joblib.load(bundle_path)
    mpath = Path(bundle["model_path"])
    mpath = mode_dir / "accept_model.cbm"
    if not mpath.exists():
        raise FileNotFoundError(f"нет .cbm: {mpath}")

    clf = CatBoostClassifier()
    clf.load_model(str(mpath))

    return {
        "mode": mode,
        "clf": clf,
        "threshold": float(bundle["threshold"]),
        "feature_names": bundle["feature_names"],
        "min_precision": bundle.get("min_precision"),
        "label": MODE_DESCR.get(mode, ""),
    }


# ---------------------------------------------------------------- features

def extract_all(docs, subdivisions):
    feats_list, corrected_list = [], []
    for d in docs:
        f, c = extract_features(d, subdivisions=subdivisions)
        feats_list.append(f)
        corrected_list.append(c)
    return feats_list, corrected_list


def feats_to_frame(feats_list, feature_names):
    X = pd.DataFrame(feats_list).fillna(0.0)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    return X[feature_names]


# ---------------------------------------------------------------- inference

def run_mode(model, X, docs, corrected_list, thr_override):
    probs = model["clf"].predict_proba(X)[:, 1]
    thr = float(thr_override) if thr_override is not None else model["threshold"]

    decisions, answers = [], []
    for doc, corr, prob in zip(docs, corrected_list, probs):
        accept = bool(prob >= thr)

        # страховка: не выдаём документ с пустыми полями
        if accept:
            fields = {fn: corr.get(fn, "") for fn in FIELD_NAMES}
            if any(not v for v in fields.values()):
                accept = False
                fields = None
        else:
            fields = None

        row = {"doc_id": doc["doc_id"],
               "decision": "accept" if accept else "reject"}
        if fields is not None:
            row["fields"] = fields

        decisions.append(accept)
        answers.append(row)

    return probs, np.array(decisions, dtype=bool), answers


# ---------------------------------------------------------------- evaluation

def evaluate(docs, decisions, corrected_list, labels):
    labels = {r["doc_id"]: label_fields(r) for r in labels}
    tp = fp = fn = tn = 0
    for doc, acc, corr in zip(docs, decisions, corrected_list):
        lab = labels.get(doc["doc_id"])
        if lab is None:
            continue
        match = all(norm(corr.get(f, "")) == norm(lab.get(f, ""))
                    for f in FIELD_NAMES)
        y_true = int(match and passport_valid(lab))
        if acc and y_true:
            tp += 1
        elif acc and not y_true:
            fp += 1
        elif not acc and y_true:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "accept": tp + fp, "n": tp + fp + fn + tn}


# ---------------------------------------------------------------- output paths

def resolve_outputs(output: Path, modes):
    m = modes[0]
    if output.suffix == ".jsonl":
        output.parent.mkdir(parents=True, exist_ok=True)
        return {m: output}
    output.mkdir(parents=True, exist_ok=True)
    return {m: output / f"answers_{m}.jsonl"}


# ---------------------------------------------------------------- main

def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Accept/reject паспортов РФ. Три обученных режима.")
    ap.add_argument("predictions", help="*_predictions.jsonl")
    ap.add_argument("output", help="answers.jsonl (для --mode) "
                                   "или директория (для --all)")
    ap.add_argument("--mode", choices=MODES, default=100)

    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--thr", type=float, default=None,
                    help="переопределить порог (только для --mode)")
    ap.add_argument("--subdivisions", default=None)
    ap.add_argument("--labels", default=None,
                    help="*_labels.jsonl — посчитать P/R на лету")
    ap.add_argument("--compare", action="store_true",
                    help="напечатать сравнительную таблицу по режимам")
    ap.add_argument("--compare-json", default=None,
                    help="сохранить сравнительную таблицу в JSON")
    args = ap.parse_args()


    pred_path = Path(args.predictions).resolve()
    if not pred_path.exists():
        raise SystemExit(f"ERROR: {pred_path} не найден")

    models_root = Path(args.models_dir).resolve() if args.models_dir \
                  else here / "models"

    # ---- справочник
    sub_path = Path(args.subdivisions) if args.subdivisions \
               else (find_subdiv_path(pred_path.parent) or
                     find_subdiv_path(find_data_dir(pred_path.parent) or
                                      Path(".")))
    subdivisions = load_subdivisions(sub_path) if sub_path else {}

    modes = [args.mode]
    out_paths = resolve_outputs(Path(args.output), modes)

    print(f"[postprocess] predictions = {pred_path}")
    print(f"[postprocess] models-dir  = {models_root}")
    print(f"[postprocess] modes       = {modes}")
    print(f"[postprocess] subdivisions= {len(subdivisions)} ({sub_path})")

    # ---- модели
    models = {}
    for m in modes:
        try:
            models[m] = load_bundle(models_root, m)
        except FileNotFoundError as e:
            print(f"[postprocess] SKIP mode {m}: {e}")
    if not models:
        raise SystemExit("[postprocess] ERROR: ни одна модель не загрузилась")
    modes = list(models.keys())

    # ---- документы и фичи (один раз)
    docs = load_jsonl(pred_path)
    feats_list, corrected_list = extract_all(docs, subdivisions)
    print(f"[postprocess] docs={len(docs)}")

    # ---- метки (опционально)
    labels = load_jsonl(args.labels) if args.labels else None

    # ---- инференс по каждому режиму
    summary = []
    for m in modes:
        model = models[m]
        X = feats_to_frame(feats_list, model["feature_names"])
        probs, decisions, answers = run_mode(
            model, X, docs, corrected_list, args.thr,
        )

        n_accept = int(decisions.sum())
        info = {
            "mode": m,
            "label": model["label"],
            "threshold": float(args.thr) if args.thr is not None
                         else model["threshold"],
            "min_precision": model["min_precision"],
            "n": len(docs),
            "accept": n_accept,
            "accept_rate": n_accept / len(docs) if docs else 0.0,
        }

        if labels is not None:
            info.update(evaluate(docs, decisions, corrected_list, labels))

        out_path = out_paths[m]
        with open(out_path, "w", encoding="utf-8") as f:
            for row in answers:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        info["out"] = str(out_path)
        summary.append(info)

        line = (f"[postprocess] mode {m:>3}: accept={n_accept:>5} "
                f"({info['accept_rate']*100:5.1f}%)")
        if labels is not None:
            line += (f"  TP={info['tp']} FP={info['fp']} "
                     f"P={info['precision']:.4f} R={info['recall']:.4f}")
        print(line + f"  -> {out_path}")

    # ---- сравнительная сводка
    if args.compare or (labels is not None and len(modes) > 1):
        print("\n[postprocess] === comparison ===")
        if labels is not None:
            print(f"  {'mode':>4} {'P>=':>5} {'thr':>8} {'accept':>7} "
                  f"{'TP':>5} {'FP':>4} {'P':>7} {'R':>7}")
            for s in summary:
                print(f"  {s['mode']:>4} {s['min_precision']:>5} "
                      f"{s['threshold']:>8.4f} {s['accept']:>7} "
                      f"{s['tp']:>5} {s['fp']:>4} "
                      f"{s['precision']:>7.4f} {s['recall']:>7.4f}")
        else:
            print(f"  {'mode':>4} {'P>=':>5} {'thr':>8} "
                  f"{'accept':>7} {'rate':>7}")
            for s in summary:
                print(f"  {s['mode']:>4} {s['min_precision']:>5} "
                      f"{s['threshold']:>8.4f} {s['accept']:>7} "
                      f"{s['accept_rate']*100:>6.1f}%")

    if args.compare_json:
        with open(args.compare_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[postprocess] compare saved -> {args.compare_json}")


if __name__ == "__main__":
    main()