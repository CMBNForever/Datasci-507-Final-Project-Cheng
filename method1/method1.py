import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Time_Wasters_on_Social_Media.csv")

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

selected_cols = [
    "Total_Time_Spent",
    "Number_of_Sessions",
    "Number_of_Videos_Watched",
    "Scroll_Rate",
    "Frequency",
    "Engagement",
    "Video_Category",
    "Watch_Reason",
    "Self_Control",
    "Satisfaction",
    "Addiction_Level",
    "ProductivityLoss"
]

profile_df = df[selected_cols].copy()

numeric_cols = [
    "Total_Time_Spent",
    "Number_of_Sessions",
    "Number_of_Videos_Watched",
    "Scroll_Rate",
    "Engagement",
    "Self_Control",
    "Satisfaction"
]

categorical_cols = [
    "Video_Category",
    "Watch_Reason"
]

# Standardize
scaler = StandardScaler()
profile_df[numeric_cols] = scaler.fit_transform(profile_df[numeric_cols])

# usage_score
profile_df["usage_score"] = profile_df[
    ["Total_Time_Spent", "Number_of_Sessions", "Number_of_Videos_Watched"]
].mean(axis=1)

# interaction_score
profile_df["interaction_score"] = profile_df[
    ["Scroll_Rate", "Engagement"]
].mean(axis=1)

# self_reg_score & self_reg_risk
profile_df["self_reg_score"] = profile_df[
    ["Self_Control", "Satisfaction"]
].mean(axis=1)

profile_df["self_reg_risk"] = -profile_df["self_reg_score"]

# content one-hot encoding
content_df = pd.get_dummies(
    profile_df[["Video_Category", "Watch_Reason"]],
    prefix=["Video_Category", "Watch_Reason"],
    drop_first=False
)

content_df = content_df.astype(int)

# final dataset
final_df = pd.concat([
    profile_df[[
        "usage_score",
        "interaction_score",
        "self_reg_risk",
        "Scroll_Rate",
        "Frequency",
        "Self_Control",
        "Addiction_Level",
        "ProductivityLoss"
    ]],
    content_df
], axis=1)

# rearrange
content_cols = content_df.columns.tolist()

final_df = final_df[
    ["usage_score", "interaction_score", "self_reg_risk",
     "Scroll_Rate", "Self_Control"]
    + content_cols
    + ["Addiction_Level", "ProductivityLoss"]
]

final_df.to_csv("behavior_profile_dataset.csv", index=False)