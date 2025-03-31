**Insurance Cost Prediction - Machine Learning Project**

📌 Project Overview
This project predicts medical insurance costs using a Kaggle dataset (1,338 records) and extends predictions to 15,000 synthetic records. It implements a comprehensive ML pipeline with 12 regression models, with XGBoost emerging as the best performer after hyperparameter optimization.

🔍 Data Sources
Training Data: Kaggle Medical Insurance Dataset (1,338 records)

Features: age, sex, bmi, children, smoker, region

Target: charges (medical costs)

Prediction Data: 15,000 synthetically generated records

⚙️ Data Preprocessing
Missing Values Check: Confirmed no missing data

Feature Engineering:
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']
Transformations:

One-Hot Encoding for categorical features (with drop='first')

Standard Scaling for numerical features

preprocessor = Pipeline([
    ('encoder', OneHotEncoder(drop='first')),
    ('scaler', StandardScaler())])
    
🧠 Models Tested
Model Type	Algorithms
Linear Models	Linear Regression, Ridge
Tree-Based	Random Forest, XGBoost, GBM, AdaBoost
Neural Networks	MLP Regressor
Distance-Based	KNN, SVR
Probabilistic	Gaussian Process

🏆 Model Performance Comparison
Sorted by RMSE (lower is better):

Model	MAE	RMSE	R²	Training Time
XGBoost	2440.07	4254.78	0.88	0.04s
Gradient Boosting	2454.47	4335.04	0.88	0.17s
Random Forest	2533.58	4496.86	0.87	0.67s
Polynomial Regression	2729.50	4551.13	0.87	0.01s
AdaBoost	3089.72	4693.24	0.86	0.14s

Key Insights:
Tree-based models dominated performance:
XGBoost achieved 12% lower MAE than linear models

Simple models (Linear Regression) trained fastest but performed poorly

🎯 XGBoost Optimization
Three-phase hyperparameter tuning:

Learning Rate & Estimators
Best: learning_rate=0.1 (RMSE: 4254.78)

Tree Structure
Optimal: max_depth=3, min_child_weight=1

Regularization
Final: gamma=0, reg_alpha=0, reg_lambda=1

Final Model Metrics:
MAE: 2440.07 → 2398.32 (1.7% improvement)
RMSE: 4254.78 → 4186.45 (1.6% improvement)
R²: 0.8834 → 0.8871

📊 Prediction Results on New Data
Distribution: Predicted charges ranged from $1,120 to $48,790
Average Prediction: $13,245 ± $9,872 (std)
Output: Saved to new_customers_predictions.xlsx

💡 Key Findings
Smoker Status was the most important feature (XGBoost feature importance)

Non-linear relationships exist (tree models outperformed linear)
Hyperparameter tuning yielded ~2% performance gain
Prediction Stability: 95% of predictions fell between $3,200-$32,000

🚀 How to Reproduce
Import insurance.cvs and new_clients.xlsx and run the code


🔮 Future Improvements

Test deep learning approaches
