# 📦 Gerekli Kütüphaneler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# 1️⃣ Veri Setinin Yüklenmesi
# ----------------------------------------------------------------------------
df = pd.read_csv("housing.csv")  # California Housing veri seti

print("✅ Veri setinin ilk 5 satırı:")
print(df.head(), "\n")

print("✅ Veri setindeki değişken isimleri:")
print(df.columns, "\n")

print("✅ Veri seti hakkında bilgi:")
print(df.info(), "\n")

print("✅ Eksik değer kontrolü:")
print(df.isnull().sum(), "\n")

print("✅ 'ocean_proximity' değişkeninin aldığı değerler:")
print(df["ocean_proximity"].unique(), "\n")

print("✅ 'ocean_proximity' değişkenindeki değer sayıları:")
print(df["ocean_proximity"].value_counts(), "\n")

print("✅ Sayısal değişkenlerin betimsel istatistikleri:")
print(df.describe(), "\n")

# ----------------------------------------------------------------------------
# 2️⃣ Sayısal Değişkenlerin Histogramları
# ----------------------------------------------------------------------------
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Sayısal Değişkenlerin Histogramları", fontsize=18)
plt.show()

# ----------------------------------------------------------------------------
# 3️⃣ 'median_income' Değişkenine Göre Tabakalı Örnekleme Hazırlığı
# ----------------------------------------------------------------------------
# Amaç: Modelin eğitim ve test setlerinde gelir dağılımının aynı olmasını sağlamak.
# Bunun için median_income değerlerini kategorilere ayıracağız.

# Gelir dağılımının histogramı
df["median_income"].hist(edgecolor="black")
plt.title("Median Income Dağılımı", fontsize=16, color="darkblue")
plt.xlabel("Median Income")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()

# Geliri 5 kategoriye ayırma
df["income_cat"] = pd.cut(
    df["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],  # Aralıklar
    labels=[1, 2, 3, 4, 5]  # Kategori etiketleri
)

print("✅ Gelir kategorilerinin dağılımı:")
print(df["income_cat"].value_counts(), "\n")

# Kategorilerin görselleştirilmesi
sns.countplot(x="income_cat", data=df, color="skyblue", order=[1, 2, 3, 4, 5])
plt.title("Gelir Kategorilerinin Dağılımı", fontsize=16, color="darkblue")
plt.xlabel("Income Category")
plt.ylabel("Gözlem Sayısı")
plt.show()

# ----------------------------------------------------------------------------
# 4️⃣ Tabakalı Örnekleme ile Eğitim/Test Ayrımı
# ----------------------------------------------------------------------------
# StratifiedShuffleSplit, eğitim ve test setinde 'income_cat' oranlarının korunmasını sağlar.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]  # Hem X hem y içerir
    strat_test_set = df.loc[test_index]

# Oranların korunduğunu kontrol edelim
print("✅ Eğitim seti income_cat oranları:")
print(strat_train_set["income_cat"].value_counts(normalize=True), "\n")

print("✅ Test seti income_cat oranları:")
print(strat_test_set["income_cat"].value_counts(normalize=True), "\n")

# ----------------------------------------------------------------------------
# 5️⃣ Coğrafi Verilerin Görselleştirilmesi
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    strat_train_set["longitude"],    # X ekseni
    strat_train_set["latitude"],     # Y ekseni
    alpha=0.4,                        # Nokta saydamlığı
    s=strat_train_set["population"] / 100,  # Nokta büyüklüğü (nüfusa göre)
    c=strat_train_set["median_house_value"], # Renk (hedef değişken)
    cmap="jet",                       # Renk paleti
    edgecolor="k",
    linewidth=0.5
)
plt.colorbar(scatter, label="Median House Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices - Coğrafi Dağılım")
plt.show()

# ----------------------------------------------------------------------------
# 6️⃣ Korelasyon Analizi
# ----------------------------------------------------------------------------
# numeric_only=True ile sadece sayısal değişkenleri alıyoruz
corr_matrix = strat_train_set.corr(numeric_only=True)

# Hedef değişken (median_house_value) ile korelasyonları büyükten küçüğe sıralama
corr_with_target = corr_matrix["median_house_value"].sort_values(ascending=False)

print("✅ 'median_house_value' ile korelasyonlar (yüksekten düşüğe):")
print(corr_with_target)

