import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select features and target
features = ['Pclass','Sex','Age','Fare', 'Embarked']
target = 'Survived'

# Handle missing values
df.fillna({'Age': df['Age'].median()}, inplace=True)
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# SPlit Data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Traing Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# Train LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Predict and evaluate
lgb_pred = lgb_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")

# Train CatBoost model
cat_features = ['Pclass', 'Sex', 'Embarked']
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)

# Predict and evaluate
cat_pred = cat_model.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, cat_pred):.4f}")

# Train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")


# Train Catboost without encoding categorical features
cat_model_native = CatBoostClassifier(cat_features=['Sex', 'Embarked'], verbose=0)
cat_model_native.fit(X_train, y_train)

# Predict and evaluate
cat_preds_native = cat_model_native.predict(X_test)
print(f"CatBoost Native Accuracy: {accuracy_score(y_test, cat_preds_native):.4f}")

