"""Trening modelu klasyfikacji zmian w piersi.
Użycie: python src/train.py --config config.yaml
"""
import argparse, json, os, random, hashlib, platform, datetime
import numpy as np
import pandas as pd
import yaml, joblib, sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    df = load_breast_cancer(as_frame=True).frame
    return df.drop(columns="target"), df["target"], df


def data_fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()[:16]


def build_model(cfg):
    m = cfg["model"]
    if m["type"] == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=m["C"], max_iter=m["max_iter"]))])
    if m["type"] == "rf":
        return Pipeline([("clf", RandomForestClassifier(n_estimators=m.get("n_estimators", 200),
                                                        max_depth=m.get("max_depth"),
                                                        random_state=cfg["seed"]))])
    raise ValueError(f"Nieznany typ modelu: {m['type']}")


def main(config_path, out_dir):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg["seed"])
    X, y, df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=cfg["seed"])
    model = build_model(cfg).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"accuracy": round(accuracy_score(y_te, pred), 4),
               "roc_auc": round(roc_auc_score(y_te, proba), 4),
               "f1": round(f1_score(y_te, pred), 4)}
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    meta = {"created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "config": cfg, "metrics": metrics, "data_fingerprint": data_fingerprint(df),
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "features": list(X.columns)}
    json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(metrics))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="artifacts")
    a = p.parse_args()
    main(a.config, a.out)
