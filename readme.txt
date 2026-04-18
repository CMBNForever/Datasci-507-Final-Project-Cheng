Project Overview

This project builds a data-driven system for modeling social media behavior and assessing user risk. It includes behavioral feature engineering, machine learning-based risk prediction, and retrieval-based recommendation using public health resources.

The repository is organized into four main components (Methods 1–4), each corresponding to a stage in the pipeline.

We start with data from users regarding various social media usage habits, in step 1 we will process this data
to be used for modeling risk and retrieving recommendations. In step 2, we will create a model to determine 
whether a user is at risk of social media addiction. In step 3, we will scrape resources from WHO, SAMHSA, and other U.S. federal resources.
Based on this corpus of documents we will retrieve relevant results for the user. Finally in step 4, we will perform an
analysis on the user data to determine what variable contributes most to the multiclass risk prediction.

Method 1: Profile processing
Method 2: User Risk Modelling
Method 3: Recommendation Retrieval
Method 4: Risk Analysis and Ablation Study

Each method will have its own folder with the code and data inside of it.

Before starting please review the required setup to download the required packages.

-
Setup:
Use Python 3.10 or newer.

Create a venv:
python3 -m venv .venv
source .venv/bin/activate

Install from requirements.txt:
pip install -U pip
pip install -r requirements.txt
-

-
Method 1: Behavioral Feature Engineering and Profile Construction

method1.py is a Python script for transforming raw social media behavior data into structured behavioral profiles for downstream modeling and retrieval.

Description:
The script loads the raw social media dataset, selects behavior-related variables, and transforms them into a structured behavioral profile representation.
It standardizes numerical features, encodes categorical variables, and constructs aggregated behavioral scores including usage_score, interaction_score, and self_reg_risk.

The final output is a structured behavioral profile dataset, where each user is represented by aggregated behavioral scores, selected behavioral signals, and one-hot encoded content features, with Addiction_Level and ProductivityLoss retained for downstream use.
This dataset serves as the foundation for both risk prediction (Method 2) and retrieval-based guidance (Method 3).

Usage:
Navigate to the folder method1 and run the python file using
python3 method1.py

When executed, the script will:
load the raw social media dataset
clean column names for consistency
select behavior-related variables and exclude irrelevant fields
standardize numerical features using z-score normalization
construct aggregated behavioral scores:
usage_score (usage intensity)
interaction_score (interaction patterns)
self_reg_risk (self-regulation risk, derived from self-control and satisfaction)
perform one-hot encoding for content-related variables (Video_Category and Watch_Reason)
combine behavioral scores, selected features, and encoded variables into a unified dataset
retain Addiction_Level and ProductivityLoss as target labels (not used as features)
output the final behavioral profile dataset as: behavior_profile_dataset.csv

Acknowledgments:
This script supports the Method 1 component of the EECS 486 final project on Impact of Social Media on Mental Health, providing the core behavioral representation used for both classification and retrieval modules.
-

-
Method 2: User Risk Modelling (MUST BE RUN AFTER METHOD 1)

XGBoost.py is a Python script for training a binary risk classifier that predicts whether a user is at risk based on engineered behavioral profile features.

Description:
The script loads the social media behavior dataset and constructs the modeling feature set from usage intensity, interaction patterns, self-regulation signals, and motivation/content variables.
It trains an XGBoost classifier to estimate a risk probability for each user and selects a decision threshold for classifying “at risk” vs “not at risk.”
To support consistent inference later, it saves the trained model, the selected threshold, and the exact feature column ordering used during training.

Usage:
Run from the project root (where XGBoost.py lives):
python3 XGBoost.py

When executed, the script will:
    load the raw social media behavior dataset
    clean column names for consistency
    construct the engineered feature table used for risk modelling
    split the data into training and evaluation sets (and/or run cross-validation depending on settings)
    train an XGBoost binary classifier to predict risk
    evaluate model performance and select a classification threshold
    save the trained artifacts:
        xgb_filter.pkl (trained model)
        xgb_threshold.pkl (decision threshold)
        xgb_feature_cols.pkl (feature column order)

Acknowledgments:
This script supports the Method 2 component of the EECS 486 final project on Impact of Social Media on Mental Health, providing the risk scoring model used by downstream demo and retrieval components.
-

-
Method 3: 

Description:
This part of the project scrapes a corpus of documents from public health and federal sources, then chunks it into small sections that can be searched. First, 
target webpages (WHO, SAMHSA, and related U.S. federal resources listed in data/knowledge/urls_who_samhsa.json) are downloaded and saved as plain text. 
Those texts are split into chunks, embedded with a sentence transformer, and stored as a searchable index. Separately, the system 
processed profile table into a natural-language query, runs cosine similarity against the index, and returns the top matching chunks as supporting 
evidence for that user. 

Usage:
Navigate to the folder method3 and run the python file using
python3 scrapeData.py (This will scrape the webpages)
python3 buildIndex.py (This will build the retrieval index)
python3 terminalDemo.py (This will run a program where you can retrieve some sample profiles/enter your own data)

When executed, the scrapeData.py script will
    scrape the pages listed in data/knowledge/urls_who_samhsa.json which include resources from WHO, SAMHSA, and other U.S. federal resources.
    These webpages will be output in plain text into data/knowledge/scraped/
When executed, the buildIndex.py script will
    Chunk the scraped text files 
    embed each chunk
    save the index (chunks, embedding matrix, and metadata). 
    This data will be output into data/knowledge/index/ for later cosine similarity search.
When executed, the terminalDemo.py script will
    Run a cli to allow the user to select and query for one of our example users.
    They will also be able to answer a questionaire to see 

-
Method 4: Multiclass Risk Analysis and Ablation Study
method4.py is a Python script for three-class social media risk prediction using behavioral features and grouped ablation analysis.

Description
The script loads a raw social media behavior dataset and an engineered behavioral profile dataset, builds a three-level risk label based on Addiction Level, 
reconstructs profile features from the raw data, trains a multiclass logistic regression model, and evaluates grouped feature importance through ablation.
Its main purpose is to examine which behavioral feature groups contribute most to multiclass risk prediction and to provide interpretable evaluation 
outputs like the classification report, confusion matrix, and ablation plot.

Usage
Navigate to the folder method4 and run the python file using
python3 method4.py

When executed, the script will:
    load the raw and profile datasets
    verify that the two datasets are row-aligned
    construct three risk tiers: low, medium, and high
    print the class distribution and a correlation-based leakage check
    rebuild behavioral profile features from raw inputs
    split the data into train and test sets using stratified sampling
    train a multiclass logistic regression model
    print evaluation metrics including accuracy, macro-F1, weighted-F1, and a classification report
    print a confusion matrix
    run grouped ablation across four feature groups: summary behavior, interaction and self-regulation, content preference, and motivation
    display a bar chart showing the macro-F1 change under each ablation setting


Acknowledgments
This script supports the method 4 part of the EECS 486 final project on Impact of Social Media on Mental Health.
-

-
Presentation Demo: (MUST BE RUN AFTER ALL METHODS COMPELTE)

Description:
For our in-class demo, we built a Streamlit web app that mirrors the functionality of the Method 3 terminal demo. It lets a user either select a sample profile or enter their own questionnaire responses, then shows a short profile summary and retrieves relevant guidance excerpts. If an API key is provided, it also generates brief, evidence-grounded recommendations based on the retrieved passages.

Usage:
Run from the project root:
streamlit run app.py

When executed, the demo will:
    launch a local webpage where the user can fill out the questionnaire and submit inputs
    construct a behavioral profile representation from the responses
    score risk if trained model files are available
    retrieve the most relevant evidence passages from the indexed guidance corpus
    generate short recommendations if ANTHROPIC_API_KEY is set (set manually for now use with caution)
-

