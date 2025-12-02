Patient readmission prediction (healthcare ML)
A practical, end-to-end machine learning project predicting hospital readmissions using structured patient data. It demonstrates clean data ingestion, exploratory analysis, feature engineering, model comparison, and clear evaluation reporting—everything a stakeholder needs to understand performance and risks at a glance.

Project overview
Goal: Predict whether a patient will be readmitted (Yes/No) based on demographics, admission details, vitals, and history.

Dataset size: 3,000 rows, 10 columns.

Target: Readmission (categorical: Yes/No).

Outcome: Trained and evaluated 7 classification models with detailed reports and confusion matrices.

Key observation: Strong class imbalance behavior—many models overpredict “No”.

Dataset and schema
Columns: Patient ID, Age, Gender, Admission Type, Length of Stay, Number of Diagnoses, Blood Pressure, Blood Sugar Levels, Previous Admissions, Readmission

Types: 7 numeric (int64), 3 categorical (object)

Quality checks: No duplicates, no missing values, no negative ages

Class balance (examples):

Gender: Male 1,555; Female 1,445

Admission type: Elective 1,563; Emergency 1,437

Target values: Yes/No

Repository structure
notebooks/ EDA, feature engineering, and model training workflow

data/ Source CSV (excluded from repo if private/large)

src/ Python utilities for preprocessing and modeling

reports/ Metrics, confusion matrices, and figures

README.md Project documentation (this file)

Tip: If you’re using Databricks, keep paths parameterized and avoid hardcoding workspace user paths.

Environment setup
Python: 3.10+

Core libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, plotly

bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
If not using requirements.txt, install directly: pandas numpy scikit-learn seaborn matplotlib plotly.

Workflow
1. Data ingestion
Load: Read CSV with pandas

Inspect: shape, dtypes, columns, head

Validate: duplicates, nulls, impossible values (e.g., negative age)

2. Exploratory data analysis
Univariate: Histograms and pies for Gender, Admission Type, Readmission

Relationships: Correlation heatmap for numeric features

3. Feature engineering
Split: X = all features, y = Readmission

Encode: OneHotEncoder on categorical (Gender, Admission Type)

Scale: Optional StandardScaler on numeric features

Holdout: Train/test split (80/20), stratified by target

4. Model training
Algorithms: Logistic Regression, Decision Tree, Random Forest, SVC, KNN, Naive Bayes, Gradient Boosting

Evaluation: Accuracy, weighted Precision, weighted Recall, weighted F1; confusion matrices and classification reports

Evaluation results
Summary metrics (test set)
Model	Accuracy	Precision (weighted)	Recall (weighted)	F1 (weighted)
Logistic Regression	0.7117	0.5065	0.7117	0.5918
Decision Tree	0.5967	0.5910	0.5967	0.5938
Random Forest	0.6983	0.5528	0.6983	0.5910
Support Vector Classifier	0.7117	0.5065	0.7117	0.5918
K-Nearest Neighbors	0.6533	0.5812	0.6533	0.6027
Naive Bayes	0.7117	0.5065	0.7117	0.5918
Gradient Boosting	0.7033	0.5955	0.7033	0.5991
Interpretation:

High accuracy but low positive-class performance: Several models predict “No” almost exclusively, inflating accuracy but failing to capture “Yes”.

Best balanced start: KNN and Gradient Boosting provide a slightly more balanced weighted F1, but still underperform on “Yes”.

Class imbalance behavior
Confusion matrices indicate near-zero recall for “Yes” in some models (e.g., Logistic Regression, SVC, Naive Bayes).

Action: Apply techniques that improve minority class detection.

How to run
python
# 1) Load data
df = pd.read_csv("data/patient_readmission.csv")

# 2) EDA (plots)
# - countplots for categorical
# - correlation heatmap for numeric

# 3) Feature engineering
# - OneHotEncoder for object columns
# - Train/test split (stratify=y)

# 4) Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
Recommendations and next steps
Target redefinition: Convert “Readmission” to binary 0/1 and confirm positive class mapping.

Class handling:

Resampling: SMOTE or class-weight adjustments (e.g., class_weight="balanced" for LR, SVC, Tree-based)

Threshold tuning: Use predicted probabilities; optimize decision threshold via PR curve or ROC (maximize F1 or recall for “Yes”)

Feature improvements:

Domain features: Interaction terms (Length of Stay × Number of Diagnoses), bins for vitals, previous admissions ratio

Temporal context: Time since last admission, seasonality, weekday vs weekend

Evaluation focus:

Class-specific metrics: Precision/recall/F1 for “Yes”

PR AUC: More informative than ROC AUC for imbalance

Cost-sensitive analysis: Weight false negatives higher (missed readmissions)

Model tuning: GridSearchCV/RandomizedSearchCV for Gradient Boosting and Random Forest; calibrate probabilities if needed.

Results communication
Primary KPI: Recall and F1 for “Yes” (readmission) to reduce missed high-risk patients.

Stakeholder dashboard: Predicted risk segments, feature importance, confusion matrix, alerts for high-risk profiles.

Operational use: Flag top-N high-risk patients for discharge planning and follow-up.
