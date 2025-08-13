import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LassoCV,RidgeCV,ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

# Veri setini programa çağırdık.
df = pd.read_csv("housing.csv")

# Veri setinin ilk 5 satırını çağırma
print("Veri setinde ilk 5 satır")
print(df.head())

# Veri setinin değişkenlerini görme
print("Veri setindeki değişkenler")
print(df.columns)

# Veri seti hakkında bilgi edinme
print("Veri setindeki değişkenlerin tipleri")
print(df.info())

# Veri setinde null değeler var mı kontrol etme
print("Veri setindeki null değeler")
print(df.isnull().sum())

# Veri setinde yer alan kategorik değişkenlerin aldığı değerlere bakma
print("Veri setinde yera alan kategorik değişken 'ocean_proximty'\'nin aldığı değerler")
print(df["ocean_proximity"].unique())

# Veri setindeki kategorik değişkenin ocean_proximity'nin içindeki değerlerin kaç kere aldığı
print("Kategorik değişken (ocean_proximity)'nin içindeki değerlerin kaç kere aldığı")
print(df["ocean_proximity"].value_counts())

# Sayısal değerlerin betimsel istatistik verilerine bakma
print("Sayıasal değerlerin betimsel istatistik sonuçları")
print(df.describe())

# Sayısal değişkenlerin histogramları
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Sayısal Değişkenlerin Histogramları", fontsize=18)
plt.show()

# numeric_only=True ile sadece sayısal değişkenleri alıyoruz
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

# İkili korelasyonlara bakma
plt.scatter(df['median_income'],df['median_house_value'],color='r')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()

# İstenilen değişkenlere göre korelasyon bakma(çoklu)
sns.pairplot(df[['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']],
             diag_kind='kde',  # histogram yerine KDE
             plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'})  # nokta boyutu ve saydamlık
plt.show()

# Heatmap ile gösterim
corr = df[['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# total_bedrooms değişkeninde 207 adet null değer var. Bunu çözmemiz lazım.
# biz burada median ile eksik değerleri doldurmaya karar verdik.
# df'in kopyasını çıkardık ve değişikliği burada yaptık.
df_yeni = df.copy()
print("total_bedrooms'un medianı = ", df_yeni["total_bedrooms"].median())
df_yeni["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
# Veri setinde null değerler var mı kontrol etme
print("Veri setindeki null değerler")
print(df_yeni.isnull().sum())

# One-Hot Encoding
df_encoded = pd.get_dummies(df_yeni, columns=['ocean_proximity'], drop_first=True)

# Scaling yapma

# Target değişkeni ayır
X = df_encoded.drop(columns=['median_house_value'])
y = df_encoded['median_house_value']

# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

"""
# benim kodlar ayrı ayrı yaptım ancak 
# numeric_cols_train ve numeric_cols_test aslında hep aynı sütun setini verir (çünkü X_train ve X_test aynı kolonlara sahip).
# Bu yüzden ikisini ayrı ayrı seçmek yerine tek sefer numeric_cols = X.select_dtypes(...) şeklinde yazıp 
# hem train hem testte kullanabiliriz. Dolayısıyla aşağıdaki yeni düzenlenmiş ve daha kısa kod var. Onu aktif bıraktım.

# Numeric kolonları seç
numeric_cols_train = X_train.select_dtypes(include=['float64']).columns
numeric_cols_test = X_test.select_dtypes(include=['float64']).columns

# Boolean kolonları seç
bool_cols_train = X_train.select_dtypes(include=['bool']).columns
bool_cols_test = X_test.select_dtypes(include=['bool']).columns

# scaler nesnesi yaratalım.
scaler = StandardScaler()

# Numeric kolonları scale et X_train için
X_scaled_numeric_train = pd.DataFrame(
    scaler.fit_transform(X_train[numeric_cols_train]),
    columns=numeric_cols_train,
    index=X_train.index
)

# Numeric kolonları scale et X_test için
X_scaled_numeric_test = pd.DataFrame(
    scaler.transform(X_test[numeric_cols_test]),
    columns=numeric_cols_test,
    index=X_test.index
)

# Boolean kolonları aynen ekle
X_final_train = pd.concat([X_scaled_numeric_train, X_train[bool_cols_train]], axis=1)
X_final_test =  pd.concat([X_scaled_numeric_test, X_test[bool_cols_test]], axis=1)

"""

numeric_cols = X.select_dtypes(include=['float64']).columns
bool_cols = X.select_dtypes(include=['bool']).columns

scaler = StandardScaler()

# Train
X_scaled_numeric_train = pd.DataFrame(
    scaler.fit_transform(X_train[numeric_cols]),
    columns=numeric_cols,
    index=X_train.index
)
X_final_train = pd.concat([X_scaled_numeric_train, X_train[bool_cols]], axis=1)

# Test
X_scaled_numeric_test = pd.DataFrame(
    scaler.transform(X_test[numeric_cols]),
    columns=numeric_cols,
    index=X_test.index
)
X_final_test = pd.concat([X_scaled_numeric_test, X_test[bool_cols]], axis=1)

regression=LinearRegression()
regression.fit(X_final_train,y_train)

## prediction
y_pred=regression.predict(X_final_test)

mse_basic=mean_squared_error(y_test,y_pred)
mae_basic=mean_absolute_error(y_test,y_pred)
rmse_basic=np.sqrt(mse_basic)
print("mse: ", mse_basic)
print("mae: ", mae_basic)
print("rmse: ", rmse_basic)

r2_basic=r2_score(y_test,y_pred)
print("r2 score: ", r2_basic)
#adjusted R-squared
print(1 - (1-r2_basic)*(len(y_test)-1)/(len(y_test)-X_final_test.shape[1]-1))

plt.scatter(y_test,y_pred)
plt.show()

residuals=y_test-y_pred
print(residuals)

## if residuals are in normal distribution it seems good
sns.histplot(residuals, kde=True)
plt.title("Residuals Dağılımı")
plt.show()

print("Regresyon sabiti = ",regression.intercept_)
print("Regresyon katsayıları = ",regression.coef_)

# lasso'ya bakalım.
lasso = Lasso()
lasso.fit(X_final_train, y_train)
y_pred = lasso.predict(X_final_test)
mae_lasso = mean_absolute_error(y_test, y_pred)
mse_lasso = mean_squared_error(y_test, y_pred)
rmse_lasso=np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae_lasso)
print("Mean Squared Error: ", mse_lasso)
print("rmse: ", rmse_lasso)
print("R2 Score: ", r2_lasso)
plt.scatter(y_test, y_pred)
plt.show()

# ridge'ye bakalım.
ridge = Ridge()
ridge.fit(X_final_train, y_train)
y_pred = ridge.predict(X_final_test)
mae_ridge = mean_absolute_error(y_test, y_pred)
mse_ridge = mean_squared_error(y_test, y_pred)
rmse_ridge=np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae_ridge)
print("Mean Squared Error: ", mse_ridge)
print("rmse: ", rmse_ridge)
print("R2 Score: ", r2_ridge)
plt.scatter(y_test, y_pred)
plt.show()

# elasticnet'e bakalım.
elasticnet = ElasticNet()
elasticnet.fit(X_final_train, y_train)
y_pred = elasticnet.predict(X_final_test)
mae_elasticnet = mean_absolute_error(y_test, y_pred)
mse_elasticnet = mean_squared_error(y_test, y_pred)
rmse_elasticnet=np.sqrt(mse_elasticnet)
r2_elasticnet = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae_elasticnet)
print("Mean Squared Error: ", mse_elasticnet)
print("rmse: ", rmse_elasticnet)
print("R2 Score: ", r2_elasticnet)
plt.scatter(y_test, y_pred)
plt.show()

# Tablo oluşturmak
results = {
    "Model": ["Linear Regression", "Lasso", "Ridge", "ElasticNet"],
    "MAE": [mae_basic, mae_lasso, mae_ridge, mae_elasticnet],
    "MSE": [mse_basic, mse_lasso, mse_ridge, mse_elasticnet],
    "RMSE": [rmse_basic, rmse_lasso, rmse_ridge, rmse_elasticnet],
    "R2": [r2_basic, r2_lasso, r2_ridge, r2_elasticnet]
}

results_df = pd.DataFrame(results)
results_df["R2 (%)"] = results_df["R2"] * 100

# Tabloyu yazdır
print("\nModel Karşılaştırma Tablosu:\n")
print(results_df.to_string(index=False, formatters={
    "MAE": "{:.2f}".format,
    "MSE": "{:.2f}".format,
    "RMSE": "{:.2f}".format,
    "R2": "{:.2f}".format,
    "R2 (%)": "{:.2f}".format
}))

from sklearn.model_selection import cross_val_score

# -----------------------
# MODELLERİ TANIMLA
# -----------------------
models = {
    "Lasso": LassoCV(cv=5, random_state=42),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    "ElasticNet": ElasticNetCV(cv=5, random_state=42)
}

results = []

for name, model in models.items():
    # Modeli eğit
    model.fit(X_final_train, y_train)

    # Tahminler
    y_pred = model.predict(X_final_test)

    # Metrikler
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation R² ortalaması
    cv_r2 = cross_val_score(model, X_final_train, y_train, cv=5, scoring='r2').mean()

    # Sonuçları ekle
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R² (Test)": r2,
        "R² (CV)": cv_r2
    })

# Sonuç tablosu
df_results = pd.DataFrame(results)
print("\n--- Model Karşılaştırma Tablosu ---\n")
print(df_results)

# Tahmin grafiği (en iyi modeli seç)
best_model_name = df_results.sort_values(by="R² (Test)", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
y_best_pred = best_model.predict(X_final_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahminler")
plt.title(f"{best_model_name} - Gerçek vs Tahmin")
plt.show()

# Hepsinin CV 'isine bakıp karşılaştırma tekrar olabilir ancak örnek olarak bulunsun diyoruz.
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

# -----------------------
# Modelleri tanımla
# -----------------------
models = {
    "Linear Regression": LinearRegression(),
    "LassoCV": LassoCV(cv=5, random_state=42),
    "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    "ElasticNetCV": ElasticNetCV(cv=5, random_state=42)
}

results = []

# 5 katlı KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Eğer Linear Regression ise pipeline ile scale uygula
    if name == "Linear Regression":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('linreg', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # CV R²
        cv_r2 = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=kf).mean()
        best_alpha = "-"
    else:
        model.fit(X_final_train, y_train)
        y_pred = model.predict(X_final_test)
        cv_r2 = cross_val_score(model, X_final_train, y_train, scoring='r2', cv=kf).mean()
        # alpha bilgisini kaydet
        if hasattr(model, 'alpha_'):
            best_alpha = model.alpha_
        else:
            best_alpha = "-"

    # Test metrikleri
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları ekle
    results.append({
        "Model": name,
        "Best Param (alpha)": best_alpha,
        "Test MAE": mae,
        "Test MSE": mse,
        "Test RMSE": rmse,
        "Test R2": r2,
        "CV R2 Mean": cv_r2
    })

# DataFrame oluştur
results_df = pd.DataFrame(results)


# Float formatlama fonksiyonu
def format_float(val):
    if isinstance(val, (int, float)) and not np.isnan(val):
        return f"{val:.2f}"
    else:
        return str(val)


# Tabloyu yazdır
print("\n--- Model Karşılaştırma Tablosu ---\n")
print(results_df.to_string(index=False, formatters={
    "Test MAE": format_float,
    "Test MSE": format_float,
    "Test RMSE": format_float,
    "Test R2": format_float,
    "CV R2 Mean": format_float
}))

# En iyi modelin tahmin grafiği
best_model_name = results_df.sort_values(by="Test R2", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

if best_model_name == "Linear Regression":
    y_best_pred = pipe.predict(X_test)
else:
    y_best_pred = best_model.predict(X_final_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahminler")
plt.title(f"{best_model_name} - Gerçek vs Tahmin")
plt.show()





















