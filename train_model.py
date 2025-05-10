import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def main(csv_path, model_out, test_size=0.4, seed=42):
    df = pd.read_csv(csv_path)
    y   = df["win"].astype(int)
    X   = df.drop(columns=["win", "match_id"]).copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:,1]

    print(classification_report(y_val, preds, digits=3))
    print("ROC-AUC :", roc_auc_score(y_val, proba).round(4))

    joblib.dump(model, model_out)
    print(f"\n✅ Saved model ➜ {model_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV produced by convert_json_to_csv.py")
    ap.add_argument("-m", "--model_out", default="xgb_win_predictor.joblib")
    args = ap.parse_args()
    main(args.csv_path, args.model_out)
