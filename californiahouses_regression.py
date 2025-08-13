import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LassoCV,RidgeCV,ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline

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
print(df["ocean_proximity"].value_counts())

# Sayısal değerlerin betimsel istatistik verilerine bakma
print("Sayıasal değerlerin betimsel istatistik sonuçları")
print(df.describe())

# Tüm sayısal değişkenlerin histogram grafiği aynı sayfada gösterim
# Kısa şekilde toplu gösterim
df.hist(bins=50, figsize=(20,15))
plt.show()

"""
# Daha güzel uzun gösterim.
# Tüm sayısal değişkenleri otomatik bul
num_vars = df.select_dtypes(include=[np.number]).columns

# Kaç sütun ve satır olacağını hesapla
n_cols = 3
n_rows = int(np.ceil(len(num_vars) / n_cols))

# Figure ve axis grid oluştur
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))

# Eğer tek satır varsa axes dizisini 2D'ye dönüştürelim
axes = np.array(axes).reshape(-1)

# Her değişken için histogram çiz
for i, var in enumerate(num_vars):
    ax = axes[i]
    sns.histplot(df[var].dropna(), bins=30, kde=True, color="skyblue", edgecolor="white", ax=ax)
    ax.set_title(f"Histogram of {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    ax.grid(False)

# Eğer eksik boş grafik alanları varsa onları kapatalım
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Tüm alt grafikler için tema ayarı ve boşluk düzeni
plt.tight_layout()
plt.show()
"""

# Train ve Test Veri Seti ile İlgili Çalışma
# Yanlılığı önlemek amacıyla rastgele örnekleme yerine yani
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
# bu komut yapısını kullanmak yerine tabakalı örnekleme yöntemi kullanılmaya karar verildi.

# median_income'a göre bir tabakalı örnekleme yapmak istiyoruz. Medyan gelirler
# 1.5 ile 6 (yani 15000$ ile 60000$) arasında yoğunlaşıyor
# Bazı medyan gelirleri ise 6'yı da geçiyor.
# Bu durumda 5 farklı gelir düzeyi oluşturulmasına karar verildi.
# Yani 0'dan 1.5 kategori 1, 1.5'tan 3'e kategori 2 şeklinde 5 kategori tanımlanacak.

# median_income'ın grafiğine tekrar bakalım.
df["median_income"].hist(edgecolor="black")  # kenar çizgisi ekledik
plt.title("Median Income Histogram", fontsize=16, color="darkblue")
plt.xlabel("Median Income", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)  # arka plan ızgarasını kapatmak istersen
plt.show()

# median_income'ı tabakalı hale getirelim.
df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Tabakaları yazdıralım.
print("Tabakaların dağılımı tablosal")
print(df["income_cat"].value_counts())

# Tabakalandırmanın grafiğini yapalım.
sns.countplot(x="income_cat", data=df, color="skyblue", order=[1,2,3,4,5])
plt.title("Income Category Distribution", fontsize=16, color="darkblue")
plt.xlabel("Income Category", fontsize=12)
plt.ylabel("Number of Districts", fontsize=12)
plt.show()

# Tabakalı Örneklemeyi Yapalım.
from sklearn.model_selection import StratifiedShuffleSplit # tabakalı için kütüphane
X = df.drop(["median_house_value"], axis=1)
y = df["median_house_value"]

# StratifiedShuffleSplit objesi oluştur
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(X, df["income_cat"]):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print("Train income_cat dağılımı:")
print(X_train["income_cat"].value_counts(normalize=True))
print("Test income_cat dağılımı:")
print(X_test["income_cat"].value_counts(normalize=True))
print("----------")

# Veri setlerini görselleştirme
# Train veri setini olası bir probleme karşı kopya edelim.
X_train_kopya = X_train.copy()

"""
# Coğrafi Veriyi Görselleştirme
X_train_kopya.plot(kind="scatter",x="latitude",y="longitude",alpha=0.1)
plt.show()
"""
# Coğrafi Veriyi Görselleştirme daha ayrıntılı
plt.figure(figsize=(10,7))
scatter = plt.scatter(
    X_train_kopya["longitude"],
    X_train_kopya["latitude"],
    alpha=0.4,
    s=X_train_kopya["population"] / 100,
    c=y_train,# median_house_value'yu kullanmak istediğimiz için
    cmap="jet",
    edgecolor="k",
    linewidth=0.5
)
plt.colorbar(scatter, label="Median House Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices Scatterplot")
plt.show()

# Korelasyonların Bulunması
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]   # Bu hem özellikleri hem hedef değişkeni içerir
    strat_test_set  = df.loc[test_index]

# Artık strat_train_set'te hem X hem y var
corr_matrix = strat_train_set.corr(numeric_only=True)

# median_house_value ile korelasyonlar
corr_with_target = corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_with_target)

# Heatmap ile korelasyon
# Korelasyon matrisi
corr = df.corr(numeric_only=True)

# Sadece median_house_value ile olan korelasyonlar
target_corr = corr[['median_house_value']].sort_values(by='median_house_value', ascending=False)



























































