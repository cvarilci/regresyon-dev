# ---------------------------
# 1. Kütüphaneler
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# 2. Veri Yükleme
# ---------------------------
df = pd.read_csv("housing.csv")

# ---------------------------
# 3. Eksik değerler ve outlier temizleme
# ---------------------------
# Hedef sütun ve sayısal sütunlar
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove("median_house_value")

# Hedef sütundaki outlierları temizleme
def remove_outliers_from_column(df, col, threshold=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold*IQR
    upper = Q3 + threshold*IQR
    return df[(df[col]>=lower) & (df[col]<=upper)].copy()

df_clean = remove_outliers_from_column(df, "median_house_value")

# total_bedrooms eksik değerleri medyan ile doldurma
df_clean["total_bedrooms"] = df_clean["total_bedrooms"].fillna(df_clean["total_bedrooms"].median())

# ocean_proximity kategorik sütunu dummies ile encode etme
df_clean = pd.get_dummies(df_clean, columns=["ocean_proximity"], drop_first=True)

# ---------------------------
# 4. Performans ölçüm fonksiyonu
# ---------------------------
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    return mae, rmse, r2

# ---------------------------
# 5. Dönüşüm öncesi model
# ---------------------------
X = df_clean.drop("median_house_value", axis=1)
y = df_clean["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

model_before = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, colsample_bytree=0.7)
model_before.fit(X_train, y_train)

y_train_pred = model_before.predict(X_train)
y_test_pred = model_before.predict(X_test)

train_mae_before, train_rmse_before, train_r2_before = evaluate_model(y_train, y_train_pred)
test_mae_before, test_rmse_before, test_r2_before = evaluate_model(y_test, y_test_pred)

print("----- Dönüşüm Öncesi -----")
print("Train RMSE:", train_rmse_before, "R2:", train_r2_before)
print("Test RMSE:", test_rmse_before, "R2:", test_r2_before)

# ---------------------------
# 6. Dönüşüm sonrası model (PowerTransformer)
# ---------------------------
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove("median_house_value")

X_num = df_clean[numeric_cols]
y = df_clean["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.3, random_state=15)

pt = PowerTransformer(method="yeo-johnson")
X_train_trans = pt.fit_transform(X_train)
X_test_trans = pt.transform(X_test)

model_after = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, colsample_bytree=0.7)
model_after.fit(X_train_trans, y_train)

y_train_pred_trans = model_after.predict(X_train_trans)
y_test_pred_trans = model_after.predict(X_test_trans)

train_mae_after, train_rmse_after, train_r2_after = evaluate_model(y_train, y_train_pred_trans)
test_mae_after, test_rmse_after, test_r2_after = evaluate_model(y_test, y_test_pred_trans)

print("----- Dönüşüm Sonrası -----")
print("Train RMSE:", train_rmse_after, "R2:", train_r2_after)
print("Test RMSE:", test_rmse_after, "R2:", test_r2_after)

# ---------------------------
# 7. Performans Karşılaştırması
# ---------------------------
comparison = pd.DataFrame({
    "Before Transformation": [train_rmse_before, test_rmse_before, train_r2_before, test_r2_before],
    "After Transformation": [train_rmse_after, test_rmse_after, train_r2_after, test_r2_after]
}, index=["Train RMSE", "Test RMSE", "Train R2", "Test R2"])

print("\n----- Performans Karşılaştırması -----")
print(comparison)

# ---------------------------
# 8. Dönüşüm Öncesi ve Sonrası Dağılımlar
# ---------------------------
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Dönüşüm sonrası DataFrame
df_transformed = pd.DataFrame(X_train_trans, columns=numeric_cols)

n_cols = len(numeric_cols)
n_rows = n_cols  # her satır bir sütun
fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 4*n_rows))
fig.suptitle("Sayısal Sütunlar Dağılımları: Önce ve Sonra", fontsize=18, fontweight="bold", y=1.02)

for i, col in enumerate(numeric_cols):
    # Önceki veri histogramı
    sns.histplot(df_clean[col], kde=True, bins=30, color="steelblue", ax=axes[i,0])
    axes[i,0].set_title(f"{col} - Before", fontsize=10, fontstyle="italic")
    
    # Dönüşüm sonrası veri histogramı
    sns.histplot(df_transformed[col], kde=True, bins=30, color="teal", ax=axes[i,1])
    axes[i,1].set_title(f"{col} - After", fontsize=10, fontstyle="italic")

plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV

# ---------------------------
# Hyperparameter Tuning (After Transformation)
# ---------------------------
xgb_param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
    "subsample": [0.5, 0.7, 1.0]
}

xgb_model = XGBRegressor()

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=20,           # kaç farklı parametre kombinasyonu denenecek
    cv=3,                # 3-fold cross validation
    scoring="r2",        # R2 score ile seçim
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train_trans, y_train)

print("----- En iyi parametreler (After Transformation + Tuning) -----")
print(random_search.best_params_)

# En iyi parametrelerle modeli eğit
best_xgb_model = random_search.best_estimator_

y_train_pred_tuned = best_xgb_model.predict(X_train_trans)
y_test_pred_tuned = best_xgb_model.predict(X_test_trans)

train_mae_tuned, train_rmse_tuned, train_r2_tuned = evaluate_model(y_train, y_train_pred_tuned)
test_mae_tuned, test_rmse_tuned, test_r2_tuned = evaluate_model(y_test, y_test_pred_tuned)

print("----- After Transformation + Hyperparameter Tuning -----")
print("Train RMSE:", train_rmse_tuned, "R2:", train_r2_tuned)
print("Test RMSE:", test_rmse_tuned, "R2:", test_r2_tuned)

# ---------------------------
# Karşılaştırma Tablosu
# ---------------------------
comparison_tuned = pd.DataFrame({
    "After Transformation (No Tuning)": [train_rmse_after, test_rmse_after, train_r2_after, test_r2_after],
    "After Transformation + Tuning": [train_rmse_tuned, test_rmse_tuned, train_r2_tuned, test_r2_tuned]
}, index=["Train RMSE", "Test RMSE", "Train R2", "Test R2"])

print("\n----- After Transformation Karşılaştırması (Tuning vs No Tuning) -----")
print(comparison_tuned)








