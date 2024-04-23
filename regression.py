import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/mnt/data/vgsales.csv')

# Handle missing values
df.dropna(inplace=True)

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature engineering: convert categorical data
categorical_features = ['Platform', 'Genre']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

# Prepare features and labels
X = df.drop(['Rank', 'Name', 'Global_Sales', 'Publisher', 'Year'], axis=1)  # Exclude non-numeric and unnecessary columns
y = df['Global_Sales']

# Data normalization
scaler = MinMaxScaler()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a modeling pipeline
linear_pipe = Pipeline([
    ('transformer', transformer),
    ('scaler', scaler),
    ('model', LinearRegression())
])

random_forest_pipe = Pipeline([
    ('transformer', transformer),
    ('scaler', scaler),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit models
linear_pipe.fit(X_train, y_train)
random_forest_pipe.fit(X_train, y_train)

# Evaluate models
linear_predictions = linear_pipe.predict(X_test)
forest_predictions = random_forest_pipe.predict(X_test)

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, linear_predictions)))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, forest_predictions)))
print("Linear Regression R^2:", r2_score(y_test, linear_predictions))
print("Random Forest R^2:", r2_score(y_test, forest_predictions))

# Cross-validation for model robustness
linear_cv_score = cross_val_score(linear_pipe, X, y, cv=5, scoring='neg_mean_squared_error')
forest_cv_score = cross_val_score(random_forest_pipe, X, y, cv=5, scoring='neg_mean_squared_error')

print("Linear Regression CV RMSE:", np.sqrt(-np.mean(linear_cv_score)))
print("Random Forest CV RMSE:", np.sqrt(-np.mean(forest_cv_score)))


# Feature Importance from Random Forest
feature_importances = random_forest_pipe.named_steps['model'].feature_importances_
features = random_forest_pipe.named_steps['transformer'].transformers_[0][1].get_feature_names_out().tolist() + ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_imp_df, x='Importance', y='Feature')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Scatter plot for Actual vs Predicted sales
plt.figure(figsize=(12, 6))
plt.scatter(y_test, linear_predictions, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, forest_predictions, alpha=0.5, color='red', label='Random Forest')
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual vs Predicted Global Sales')
plt.legend()
plt.show()

# Residual Plots for models
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.residplot(x=y_test, y=linear_predictions, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residuals for Linear Regression')
plt.xlabel('Actual Global Sales')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
sns.residplot(x=y_test, y=forest_predictions, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residuals for Random Forest')
plt.xlabel('Actual Global Sales')
plt.ylabel('Residuals')
plt.show()