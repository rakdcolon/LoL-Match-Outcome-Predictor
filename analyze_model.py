import argparse, joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from xgboost import plot_importance


def main(csv_path, model_path):
    # Load data and model
    df = pd.read_csv(csv_path)
    y = df["win"].astype(int)
    X   = df.drop(columns=["win", "match_id"]).copy()

    # Load model
    model = joblib.load(model_path)

    # Predict
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, preds, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Feature Importance
    plot_importance(model, max_num_features=20)
    plt.title("Top 20 Feature Importances (XGBoost)")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV produced by convert_json_to_csv.py")
    ap.add_argument("model_path", help="Trained XGBoost model (joblib)")
    args = ap.parse_args()
    main(args.csv_path, args.model_path) 