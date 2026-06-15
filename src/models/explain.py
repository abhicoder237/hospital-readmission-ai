# src/models/explain.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model_and_data():
    """Saved model aur test data load karta hai."""
    print("📂 Loading model and data...")

    model        = joblib.load('models/readmission_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    X_test       = pd.read_csv('data/processed/X_test.csv')
    y_test       = pd.read_csv('data/processed/y_test.csv').squeeze()

    print(f"   ✅ Model loaded")
    print(f"   ✅ Test data: {X_test.shape}")

    return model, X_test, y_test, feature_cols


def generate_shap_values(model, X_test):
    """
    SHAP values calculate karta hai.

    SHAP = SHapley Additive exPlanations
    Har feature ka contribution calculate karta hai
    final prediction mein.

    Example:
    Base prediction = 0.25 (average readmission rate)
    + Age=75        → +0.12 (pushes risk UP)
    + Glucose=180   → +0.08 (pushes risk UP)
    + n_diagnoses=1 → -0.05 (pushes risk DOWN)
    = Final score   → 0.40
    """
    print("\n🔍 Calculating SHAP values...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print(f"   ✅ SHAP values shape: {shap_values.shape}")
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, save_path):
    """
    SHAP Summary Plot — sabse important visualization.
    Dikhata hai:
    - Kaunse features sabse important hain (y-axis)
    - High/low feature value ka kya effect (color)
    - Effect kitna bada hai (x-axis)
    """
    print("\n🎨 Generating SHAP plots...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",
        max_display=15,
        show=False
    )
    plt.title(
        'SHAP Feature Impact on Readmission Risk',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_shap_bar(shap_values, X_test, save_path):
    """
    SHAP Bar Plot — average impact per feature.
    Simple aur doctor-friendly.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title(
        'Average Feature Importance (SHAP)',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def explain_single_patient(model, explainer, X_test,
                           y_test, feature_cols,
                           patient_idx=0):
    """
    Ek specific patient ki prediction explain karta hai.
    Doctor ko yeh dikhayenge API response mein.
    """
    print(f"\n👤 Explaining Patient #{patient_idx}...")

    patient_data = X_test.iloc[[patient_idx]]
    shap_vals    = explainer.shap_values(patient_data)[0]
    pred_prob    = model.predict_proba(patient_data)[0][1]
    actual       = y_test.iloc[patient_idx]

    print(f"\n   Predicted Risk : {pred_prob:.1%}")
    print(f"   Actual         : {'Readmitted' if actual else 'Not Readmitted'}")
    print(f"   Risk Level     : {'🔴 HIGH' if pred_prob > 0.5 else '🟡 MEDIUM' if pred_prob > 0.3 else '🟢 LOW'}")

    # Top factors
    feature_impact = pd.Series(
        shap_vals,
        index=feature_cols
    ).sort_values(key=abs, ascending=False)

    print(f"\n   Top 5 factors:")
    print(f"   {'Feature':<25} {'Impact':<10} Direction")
    print(f"   {'-'*50}")

    for feat, impact in feature_impact.head(5).items():
        direction = "⬆️ INCREASES risk" if impact > 0 else "⬇️ DECREASES risk"
        print(f"   {feat:<25} {impact:+.4f}   {direction}")

    # Waterfall plot — single patient
    save_path = f'models/shap_patient_{patient_idx}.png'

    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_vals,
            base_values   = explainer.expected_value,
            data          = patient_data.iloc[0].values,
            feature_names = feature_cols
        ),
        max_display = 10,
        show        = False
    )
    plt.title(
        f'Patient #{patient_idx} — Risk Explanation',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: {save_path}")

    return {
        'patient_idx':  patient_idx,
        'predicted_risk': round(float(pred_prob), 4),
        'risk_level':   ('HIGH' if pred_prob > 0.5
                         else 'MEDIUM' if pred_prob > 0.3
                         else 'LOW'),
        'top_factors':  feature_impact.head(5).to_dict(),
        'actual':       int(actual)
    }


def run_explainability():
    """Poora SHAP pipeline chalata hai."""

    print("=" * 55)
    print("🔍 SHAP Explainability Pipeline")
    print("=" * 55)

    # Load
    model, X_test, y_test, feature_cols = load_model_and_data()

    # SHAP values
    explainer, shap_values = generate_shap_values(model, X_test)

    # Plots
    os.makedirs('models', exist_ok=True)
    plot_shap_summary(shap_values, X_test,
                      'models/shap_summary.png')
    plot_shap_bar(shap_values, X_test,
                  'models/shap_bar.png')

    # Single patient examples — High risk aur Low risk
    print("\n" + "─" * 55)
    print("📋 Individual Patient Explanations")
    print("─" * 55)

    # High risk patient dhundho
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(y_pred_prob)
    low_risk_idx  = np.argmin(y_pred_prob)

    explain_single_patient(
        model, explainer, X_test, y_test,
        feature_cols, patient_idx=int(high_risk_idx)
    )
    explain_single_patient(
        model, explainer, X_test, y_test,
        feature_cols, patient_idx=int(low_risk_idx)
    )

    print("\n" + "=" * 55)
    print("✅ Explainability Complete!")
    print("=" * 55)
    print("\n📁 Generated files:")
    print("   models/shap_summary.png")
    print("   models/shap_bar.png")
    print("   models/shap_patient_*.png")


if __name__ == "__main__":
    run_explainability()