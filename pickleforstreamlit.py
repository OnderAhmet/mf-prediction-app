import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Örnek veri oluşturma
df = pd.read_excel("ML_data_cleaned_all.xlsx")  # Gerçek veri dosyanızı kullanın

# Kullanılacak sütunları belirleme
numeric_columns = ['Mud Weight, ppg', 'Yield Point, lbf/100 sqft', 'Chlorides, mg/L',
                   'Solids, %vol', 'HTHP Fluid Loss, cc/30min', 'pH',
                   'NaCl (SA), %vol', 'KCl (SA), %vol', 'Low Gravity (SA), %vol',
                   'Drill Solids (SA), %vol', 'R600', 'R300', 'R200', 'R100', 'R6', 'R3', 'Average SG Solids (SA)']

df = df.dropna()  # Eksik verileri kaldırma

# Özellikler ve hedef değişkenleri tanımlama
X = df[numeric_columns]
y_mf = df['Mf'].values
y_pf = df['Pf'].values
y_pm = df['Alkal Mud (Pm)'].values

# Veriyi ölçeklendirme
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns)
    ])

X_transformed = preprocessor.fit_transform(X)

# Mf tahmini için Gradient Boosting modeli
gb_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=12,
    n_estimators=450,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_transformed, y_mf)

# Pf tahmini için CatBoost modeli
catboost_model = CatBoostRegressor(depth=8, iterations=100, learning_rate=0.05, verbose=0)
catboost_model.fit(y_mf.reshape(-1, 1), y_pf)

# Pm tahmini için XGBoost modeli
xgb_model = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05)
xgb_model.fit(y_mf.reshape(-1, 1), y_pm)

# Modelleri pickle olarak kaydetme
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("gb_model.pkl", "wb") as f:
    pickle.dump(gb_model, f)

with open("catboost_model.pkl", "wb") as f:
    pickle.dump(catboost_model, f)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("Model eğitimleri tamamlandı ve pickle dosyaları oluşturuldu!")
