import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)


# ── 1.load ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / 'data' / 'behavior_profile_dataset.csv')

print("数据维度:", df.shape)
print("\n标签分布:\n", df['risk_binary'].value_counts())

# ── 2. separate features and labels ──────────────────────────────────────
drop_cols = ['risk_binary', 'Addiction_Level', 
             'Self_Control', 'ProductivityLoss', 'self_reg_risk']

X = df.drop(columns=drop_cols)
y = df['risk_binary']


# ── 3. leak check ─────────────────────────────
corr = df.corr()['risk_binary'].abs().sort_values(ascending=False)
print(corr)

# ── 3. test set ─────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size    = 0.2,       # 80% train，20% test
    stratify     = y,         # same proportion
    random_state = 42
)

print(f"\ntrain set size: {X_train_full.shape[0]}")
print(f"test set size: {X_test.shape[0]}")

# ── 4. model ────────────────────────────────────────────
neg, pos = (y_train_full == 0).sum(), (y_train_full == 1).sum()

model = xgb.XGBClassifier(
    n_estimators          = 500,
    learning_rate         = 0.05,
    max_depth             = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    scale_pos_weight      = neg / pos,
    eval_metric           = 'auc',
    early_stopping_rounds = 50,
    random_state          = 42,
    verbosity             = 0
)

# ── 5. cross-validation ────────────────────────────────
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train_full))
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):

    X_tr,  X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr,  y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    model.fit(
        X_tr, y_tr,
        eval_set = [(X_val, y_val)],
        verbose  = False
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_aucs.append(fold_auc)
    print(f"Fold {fold+1}  AUC: {fold_auc:.4f}")

print(f"\nCV Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# ── 6. final model ───────────────────────
X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
    X_train_full, y_train_full,
    test_size    = 0.1,
    stratify     = y_train_full,
    random_state = 42
)

model.fit(
    X_tr_final, y_tr_final,
    eval_set = [(X_val_final, y_val_final)],
    verbose  = False
)

# ── 7. evaluation ──────────────────────────
test_preds = model.predict_proba(X_test)[:, 1]
test_auc   = roc_auc_score(y_test, test_preds)
y_pred     = (test_preds > 0.5).astype(int)

print(f"\nTest AUC: {test_auc:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# ── 8. threshold ────────────────────────────────────────────
# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, test_preds)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

# ── 8b. Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, test_preds)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=150)
plt.show()

min_recall = 0.90
valid_idx  = np.where(recall[:-1] >= min_recall)[0]
best_idx   = valid_idx[np.argmax(precision[valid_idx])]
best_threshold = pr_thresholds[best_idx]

print(f"\nbest threshold: {best_threshold:.4f}")
print(f"for Precision: {precision[best_idx]:.4f}")
print(f"for Recall:    {recall[best_idx]:.4f}")
print(f"high risk user proportion: {(test_preds > best_threshold).mean():.2%}")

y_pred_best = (test_preds > best_threshold).astype(int)
print("\nClassification Report (best threshold):")
print(classification_report(y_test, y_pred_best,
                             target_names=['Low Risk', 'High Risk']))

# confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low Risk', 'High Risk'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

joblib.dump(model, 'xgb_filter.pkl')
joblib.dump(best_threshold, 'xgb_threshold.pkl')
joblib.dump(list(X.columns), 'xgb_feature_cols.pkl')

print(f"\n model saved: xgb_filter.pkl  threshold: {best_threshold:.4f}")

if shap is not None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

# rank
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150)
plt.show()

# direction
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
plt.show()

def should_retrieve(user_features: dict) -> bool:
    """
    To determine whether need to retrieve.
    user_features: dictionary that same as the training columns
        {'Daily_Usage_Time': 3.5, 'Posts_Per_Day': 2, ...}
    True means high risk，need retrieve；False means no retrieve needed
    """
    _model     = joblib.load('xgb_filter.pkl')
    _threshold = joblib.load('xgb_threshold.pkl')
    _cols      = joblib.load('xgb_feature_cols.pkl')

    row  = pd.DataFrame([user_features])[_cols]   
    prob = _model.predict_proba(row)[0, 1]
    return bool(prob >= _threshold)
