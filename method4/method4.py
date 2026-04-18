import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

raw_df = pd.read_csv("Time_Wasters_on_Social_Media.csv")
profile_df = pd.read_csv("behavior_profile_dataset.csv")

print("raw_df shape:", raw_df.shape)
print("profile_df shape:", profile_df.shape)

assert np.array_equal(profile_df["Addiction_Level"].values, raw_df["Addiction Level"].values)
assert np.array_equal(profile_df["ProductivityLoss"].values, raw_df["ProductivityLoss"].values)

analysis_df = profile_df.copy()
analysis_df["UserID"] = raw_df["UserID"].values


analysis_df["Addiction_Level_raw"] = raw_df["Addiction Level"].values
analysis_df["ProductivityLoss_raw"] = raw_df["ProductivityLoss"].values
# Risk is defined with 3 level (low: 0-2, medium: 3-4, high: 5-7)
analysis_df["risk_tier"] = pd.cut(
    analysis_df["Addiction_Level_raw"],
    bins=[-1, 2, 4, 7],
    labels=["low", "medium", "high"]
)
print("\nRisk tier distribution:")
print(analysis_df["risk_tier"].value_counts().sort_index())
print("\nCorrelation check:")
print(
    raw_df[["Addiction Level", "ProductivityLoss", "Self Control"]]
    .corr(numeric_only=True)
)

# define feature groups for ablation
content_cols = [c for c in analysis_df.columns if c.startswith("Video_Category_")]
motivation_cols = [c for c in analysis_df.columns if c.startswith("Watch_Reason_")]

feature_groups = {
    "summary_behavior": [
        "usage_score"
    ],
    "interaction_self_regulation": [
        "interaction_score",
        "self_reg_risk",
        "Scroll_Rate"
    ],
    "content_preference": content_cols,
    "motivation": motivation_cols
}
print("\nFeature groups:")
for k, v in feature_groups.items():
    print(f"{k}: {len(v)} columns")

RAW_NUMERIC_COLS = [
    "Total Time Spent",
    "Number of Sessions",
    "Number of Videos Watched",
    "Scroll Rate",
    "Engagement",
    "Self Control",
    "Satisfaction"
]

def fit_profile_stats(raw_train_df):
    """
    Learn all training-only stats needed to process future raw users.
    """
    stats = {}

    for col in RAW_NUMERIC_COLS:
        stats[col] = {
            "mean": raw_train_df[col].mean(),
            "std": raw_train_df[col].std(ddof=0)
        }

    stats["scroll_low"] = raw_train_df["Scroll Rate"].quantile(0.33)
    stats["scroll_high"] = raw_train_df["Scroll Rate"].quantile(0.67)
    stats["video_categories"] = sorted(raw_train_df["Video Category"].dropna().unique().tolist())
    stats["watch_reasons"] = sorted(raw_train_df["Watch Reason"].dropna().unique().tolist())

    return stats


def safe_zscore(series, mean, std):
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def build_profile_from_raw(raw_input_df, stats):
    """
    Convert future raw user input into the same profile feature space as training.
    raw_input_df should contain the original raw columns, e.g.
    Total Time Spent, Number of Sessions, Number of Videos Watched, Scroll Rate,
    Engagement, Video Category, Watch Reason, Self Control, Satisfaction.
    """
    df = raw_input_df.copy()

    z_total_time = safe_zscore(df["Total Time Spent"], stats["Total Time Spent"]["mean"], stats["Total Time Spent"]["std"])
    z_sessions = safe_zscore(df["Number of Sessions"], stats["Number of Sessions"]["mean"], stats["Number of Sessions"]["std"])
    z_videos = safe_zscore(df["Number of Videos Watched"], stats["Number of Videos Watched"]["mean"], stats["Number of Videos Watched"]["std"])
    z_scroll = safe_zscore(df["Scroll Rate"], stats["Scroll Rate"]["mean"], stats["Scroll Rate"]["std"])
    z_engagement = safe_zscore(df["Engagement"], stats["Engagement"]["mean"], stats["Engagement"]["std"])
    z_self_control = safe_zscore(df["Self Control"], stats["Self Control"]["mean"], stats["Self Control"]["std"])
    z_satisfaction = safe_zscore(df["Satisfaction"], stats["Satisfaction"]["mean"], stats["Satisfaction"]["std"])

    profile = pd.DataFrame(index=df.index)
    profile["usage_score"] = (z_total_time + z_sessions + z_videos) / 3
    profile["interaction_score"] = (z_scroll + z_engagement) / 2
    profile["self_reg_risk"] = -(z_self_control + z_satisfaction) / 2
    profile["Scroll_Rate"] = z_scroll

    for cat in stats["video_categories"]:
        profile[f"Video_Category_{cat}"] = (df["Video Category"] == cat).astype(int)
    for reason in stats["watch_reasons"]:
        profile[f"Watch_Reason_{reason}"] = (df["Watch Reason"] == reason).astype(int)

    profile["Video Category"] = df["Video Category"]
    profile["Watch Reason"] = df["Watch Reason"]

    return profile

all_feature_cols = []
for cols in feature_groups.values():
    all_feature_cols.extend(cols)

# remove duplicates just in case
all_feature_cols = list(dict.fromkeys(all_feature_cols))

print("Total number of features:", len(all_feature_cols))
print(all_feature_cols)

raw_train_df, raw_test_df, profile_train_df, profile_test_df = train_test_split(
    raw_df,
    analysis_df,
    test_size=0.2,
    random_state=42,
    stratify=analysis_df["risk_tier"]
)

profile_stats = fit_profile_stats(raw_train_df)
train_profile = build_profile_from_raw(raw_train_df, profile_stats)
test_profile = build_profile_from_raw(raw_test_df, profile_stats)

train_profile["risk_tier"] = profile_train_df["risk_tier"].values
test_profile["risk_tier"] = profile_test_df["risk_tier"].values

train_profile["UserID"] = profile_train_df["UserID"].values
test_profile["UserID"] = profile_test_df["UserID"].values

train_profile["Addiction_Level_raw"] = profile_train_df["Addiction_Level_raw"].values
train_profile["ProductivityLoss_raw"] = profile_train_df["ProductivityLoss_raw"].values
test_profile["Addiction_Level_raw"] = profile_test_df["Addiction_Level_raw"].values
test_profile["ProductivityLoss_raw"] = profile_test_df["ProductivityLoss_raw"].values


X_train = train_profile[all_feature_cols].copy()
y_train = train_profile["risk_tier"].copy()

X_test = test_profile[all_feature_cols].copy()
y_test = test_profile["risk_tier"].copy()

idx_train = train_profile.index
idx_test = test_profile.index

def fit_and_evaluate(feature_list, model_name):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])

    model.fit(X_train[feature_list], y_train)
    pred = model.predict(X_test[feature_list])
    proba = model.predict_proba(X_test[feature_list])

    result = {
        "model": model_name,
        "n_features": len(feature_list),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted")
    }

    return model, pred, proba, result

full_model, full_pred, full_proba, full_result = fit_and_evaluate(
    all_feature_cols,
    "full_model"
)

print(full_result)
print("\nClassification report:")
print(classification_report(y_test, full_pred))


# confusion matrix
labels_order = ["low", "medium", "high"]

cm = confusion_matrix(y_test, full_pred, labels=labels_order)
cm_df = pd.DataFrame(cm, 
                     index=[f"true_{x}" for x in labels_order],
                     columns=[f"pred_{x}" for x in labels_order])
print(cm_df)


# ablation
ablation_results = []
ablation_results.append(full_result)

for group_name, cols in feature_groups.items():
    reduced_features = [c for c in all_feature_cols if c not in cols]

    _, _, _, result = fit_and_evaluate(
        reduced_features,
        f"without_{group_name}"
    )
    ablation_results.append(result)

ablation_df = pd.DataFrame(ablation_results)

full_macro_f1 = ablation_df.loc[
    ablation_df["model"] == "full_model", "macro_f1"
].iloc[0]

ablation_df["macro_f1_drop"] = full_macro_f1 - ablation_df["macro_f1"]

print(ablation_df.sort_values("macro_f1_drop", ascending=False))


# plot for ablation results
plot_df = ablation_df[ablation_df["model"] != "full_model"].copy()
plot_df = plot_df.sort_values("macro_f1_drop", ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(plot_df["model"], plot_df["macro_f1_drop"])
plt.ylabel("Drop in Macro-F1")
plt.xlabel("Ablation setting")
plt.title("Method 4 Ablation Study")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
