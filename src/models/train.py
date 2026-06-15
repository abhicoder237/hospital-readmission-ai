 # src/models/train.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# STEP 1: Data Load
# ============================================
def load_data():
    print("📂 Loading preprocessed data...")

    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test  = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Train class dist: {y_train.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


# ============================================
# STEP 2: SMOTE — Class Imbalance Fix
# ============================================
def apply_smote(X_train, y_train):
    """
    SMOTE = Synthetic Minority Oversampling Technique
    Minority class ke synthetic samples banata hai.
    SIRF train data pe apply karo — test pe kabhi nahi!
    """
    print("\n⚖️  Applying SMOTE...")
    print(f"   Before: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Proper pandas Series banao
    y_resampled = pd.Series(y_resampled, name='readmitted')

    print(f"   After : {y_resampled.value_counts().to_dict()}")
    print(f"   New train size: {len(X_resampled)}")

    return X_resampled, y_resampled


# ============================================
# STEP 3: Model Training
# ============================================
def train_model(X_train, y_train, params=None):
    """XGBoost model train karta hai."""

    if params is None:
        params = {
            'n_estimators':     300,
            'max_depth':        3,
            'learning_rate':    0.05,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma':            0.1,
            'reg_alpha':        0.1,
            'reg_lambda':       1.0,
            'random_state':     42,
            'eval_metric':      'auc',
        }

    print("\n🤖 Training XGBoost model...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    print("   ✅ Training complete!")
    return model, params


# ============================================
# STEP 4: Evaluation with Threshold Tuning
# ============================================
def evaluate_model(model, X_test, y_test, feature_cols):
    """
    Model evaluate karta hai + best threshold dhundta hai.
    Default 0.5 threshold healthcare ke liye sahi nahi —
    kyunki recall zyada important hai.
    """
    print("\n📊 Evaluating model...")

    # Probability scores
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # ── Best Threshold dhundho ────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, y_pred_prob
    )
    f1_scores = (
        2 * (precisions * recalls)
        / (precisions + recalls + 1e-8)
    )
    best_idx       = np.argmax(f1_scores)
    best_threshold = (
        thresholds[best_idx]
        if best_idx < len(thresholds)
        else 0.5
    )
    print(f"   🎯 Best threshold: {best_threshold:.3f}")

    # Best threshold se final predictions
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # ── Metrics ──────────────────────────────────────
    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred),          4),
        'precision': round(precision_score(y_test, y_pred,
                           zero_division=0),                         4),
        'recall':    round(recall_score(y_test, y_pred,
                           zero_division=0),                         4),
        'f1':        round(f1_score(y_test, y_pred,
                           zero_division=0),                         4),
        'roc_auc':   round(roc_auc_score(y_test, y_pred_prob),      4),
        'threshold': round(best_threshold,                           4),
    }

    print(f"\n   📈 METRICS (threshold={best_threshold:.2f}):")
    print(f"   Accuracy  : {metrics['accuracy']  * 100:.1f}%")
    print(f"   Precision : {metrics['precision'] * 100:.1f}%")
    print(f"   Recall    : {metrics['recall']    * 100:.1f}%")
    print(f"   F1 Score  : {metrics['f1']        * 100:.1f}%")
    print(f"   ROC-AUC   : {metrics['roc_auc']   * 100:.1f}%")

    print("\n   📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Not Readmitted', 'Readmitted'],
        zero_division=0
    ))

    return metrics, y_pred, y_pred_prob


# ============================================
# STEP 5: Plots
# ============================================
def plot_confusion_matrix(y_test, y_pred, save_path):
    """Confusion matrix banata hai."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues', ax=ax,
        xticklabels=['Not Readmitted', 'Readmitted'],
        yticklabels=['Not Readmitted', 'Readmitted'],
        linewidths=0.5
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5, -0.15,
        f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
        transform=ax.transAxes,
        ha='center', fontsize=10, color='gray'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")
    return cm


def plot_feature_importance(model, feature_cols, save_path):
    """Top 15 important features plot karta hai."""
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(
        kind='barh', ax=ax,
        color='#3498db', alpha=0.85
    )
    ax.set_title(
        'Top 15 Feature Importances (XGBoost)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Importance Score')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_roc_curve(y_test, y_pred_prob, roc_auc, save_path):
    """ROC Curve plot karta hai."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        fpr, tpr,
        color='#e74c3c', lw=2,
        label=f'ROC Curve (AUC = {roc_auc:.2f})'
    )
    ax.plot(
        [0, 1], [0, 1],
        color='gray', lw=1,
        linestyle='--', label='Random (AUC = 0.50)'
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


# ============================================
# STEP 6: MLflow Experiment
# ============================================
def run_experiment(experiment_name="hospital-readmission"):
    """
    Poora training pipeline MLflow ke saath.
    Har run automatically track hota hai.
    """

    # MLflow — SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    print("=" * 55)
    print("🚀 Starting MLflow Experiment")
    print("=" * 55)

    # ── Data ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_data()
    feature_cols = list(X_train.columns)

    # ── SMOTE ────────────────────────────────────────
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── MLflow Run ───────────────────────────────────
    with mlflow.start_run(run_name="xgboost_v2_threshold_tuned"):

        # Train
        model, params = train_model(X_train_sm, y_train_sm)

        # Evaluate
        metrics, y_pred, y_pred_prob = evaluate_model(
            model, X_test, y_test, feature_cols
        )

        # ── Plots ────────────────────────────────────
        os.makedirs('models', exist_ok=True)
        cm_path  = 'models/confusion_matrix.png'
        fi_path  = 'models/feature_importance.png'
        roc_path = 'models/roc_curve.png'

        print("\n🎨 Generating plots...")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_feature_importance(model, feature_cols, fi_path)
        plot_roc_curve(
            y_test, y_pred_prob,
            metrics['roc_auc'], roc_path
        )

        # ── MLflow Log ───────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(roc_path)
        mlflow.xgboost.log_model(model, "xgboost_model")

        # ── Local Save ───────────────────────────────
        joblib.dump(model, 'models/readmission_model.pkl')
        joblib.dump(feature_cols, 'models/feature_cols.pkl')

        print("\n💾 Saved:")
        print("   models/readmission_model.pkl")
        print("   models/feature_cols.pkl")
        print("   models/confusion_matrix.png")
        print("   models/feature_importance.png")
        print("   models/roc_curve.png")

        run_id = mlflow.active_run().info.run_id
        print(f"\n🔬 MLflow Run ID: {run_id}")

    print("\n" + "=" * 55)
    print("✅ Experiment Complete!")
    print("=" * 55)
    print("\n📊 View results:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("   Open: http://localhost:5000")

    return model, metrics, feature_cols


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    model, metrics, feature_cols = run_experiment()