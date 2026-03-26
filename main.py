import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

# hide unnecessary warnings
warnings.filterwarnings('ignore')

# load the telco churn dataset from kaggle
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# clean totalcharges by converting strings to numbers and filling missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# remove the customer id column as it is not predictive
df.drop('customerID', axis=1, inplace=True)

# encode categorical text features into numbers for the model
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# separate the features from the target variable
x = df.drop('Churn', axis=1)
y = df['Churn']

# split the data into 80 percent training and 20 percent testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train the xgboost classifier
model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1, 
    random_state=42,
    eval_metric='logloss'
)
model.fit(x_train, y_train)

# calculate shap values to interpret the model decisions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)

# create and save the global feature importance plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, x_test, show=False)
plt.title("global feature impact on churn")
plt.savefig('global_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# create and save the individual prediction waterfall plot
exp = shap.Explanation(
    values=shap_values[0], 
    base_values=explainer.expected_value, 
    data=x_test.iloc[0], 
    feature_names=x_test.columns.tolist()
)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(exp, show=False)
plt.title("individual prediction breakdown")
plt.savefig('local_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

# finish execution
print("execution complete. images saved.")