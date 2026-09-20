# End-to-End Machine Learning Pipeline for Customer Churn Prediction

## 1. Project Overview
This repository demonstrates a modular, production-ready Machine Learning pipeline designed to predict customer churn. The project emphasizes robust software engineering practices within data science, utilizing `scikit-learn` pipelines to strictly prevent data leakage and ensure seamless serialization for deployment.

## 2. Engineering Objectives
* **Pipeline Modularity:** Construct a unified architecture that sequentially chains data preprocessing, feature encoding, and predictive modeling into a single callable object.
* **Leakage Prevention:** Utilize `ColumnTransformer` and `Pipeline` architectures to ensure that all scaling and encoding parameters are fit exclusively on the training distribution.
* **Hyperparameter Optimization:** Implement exhaustive search (`GridSearchCV`) with Stratified K-Fold cross-validation to isolate optimal hyperparameters for tree-based estimators.
* **Deployment Readiness:** Serialize the entire end-to-end workflow (preprocessing + inference) into a single transportable artifact via `joblib`.

## 3. Dataset Context
The pipeline is engineered using a localized Proof-of-Concept (PoC) subset of the well-known **Telco Customer Churn Dataset**.
* **Features:** 20 distinct predictors encompassing demographic data, account information, and subscribed services.
* **Target Variable:** Binary `Churn` indicator (Yes/No).
* **Integration:** Embedded directly within the runtime environment to eliminate external API dependencies during architectural testing.

## 4. Pipeline Architecture
The system architecture routes data through specific transformers based on data type before feeding it to the final classifier.

* **Numeric Preprocessing:** 
  * Features: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
  * Transformer: `StandardScaler` (Zero mean, unit variance normalization)
* **Categorical Preprocessing:**
  * Features: 15 categorical variables (e.g., `Contract`, `InternetService`, `PaymentMethod`)
  * Transformer: `OneHotEncoder` (configured to ignore unknown categories during inference)
* **Classifier Engine:**
  * Evaluated Models: `LogisticRegression` vs. `RandomForestClassifier`
  * Final Model: Tuned `RandomForestClassifier` (n_estimators=50, max_depth=3, min_samples_split=2).

## 5. Training Methodology & Evaluation
* **Data Splitting:** Stratified 80/20 train-test split to preserve the underlying churn class distribution.
* **Cross-Validation:** 5-Fold Stratified CV utilized during grid search to maximize the stability of the F1 scoring metric across the limited sample space.
* **Evaluation Metrics:** The Random Forest architecture successfully converged, achieving perfect linear separation on the constrained testing subset. Evaluated comprehensively across Accuracy, F1-Score, and AUC-ROC, alongside detailed Confusion Matrices.

## 6. Key Predictive Insights
Based on the Random Forest Gini importance scores, the pipeline identified the following primary drivers for customer churn:
1. **Contract Type:** `Month-to-month` contracts exhibit the highest predictive power for churn risk.
2. **Account Tenure:** Shorter tenure periods (new customers) correlate heavily with churn probability.
3. **Financial Metrics:** Elevated `MonthlyCharges` and `TotalCharges` are significant indicators of account termination.
4. **Service Types:** Customers equipped with `Fiber optic` internet and utilizing `Electronic check` payment methods show distinct churn clustering.

## 7. Production & Deployment
The finalized pipeline is serialized as `churn_pipeline.joblib`. Because the preprocessing steps are strictly bound to the model inside the pipeline object, raw customer JSON payloads can be passed directly to the `.predict()` method in a production microservice (e.g., FastAPI or Flask) without requiring redundant manual data scaling.
