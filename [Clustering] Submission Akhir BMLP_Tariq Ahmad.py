import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import joblib

# Load data

### MULAI CODE ###

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv"
df = pd.read_csv(url)

### SELESAI CODE ###

# Tampilkan 5 baris pertama dengan function head.

### MULAI CODE ###

head = df.head()
# print(head)

### SELESAI CODE ###

# Tinjau jumlah baris kolom dan jenis data dalam dataset dengan info.

### MULAI CODE ###

info = df.info()
# print(info)

### SELESAI CODE ###

# Menampilkan statistik deskriptif dataset dengan menjalankan describe

### MULAI CODE ###

describe = df.describe()
# print(describe)

### SELESAI CODE ###

# Menampilkan korelasi antar fitur (Opsional Skilled 1)

# Memilih kolom numerik
numerical_cols = df.select_dtypes(include=["number"]).columns

### MULAI CODE ###

# Hitung matriks korelasi
correlation = df[numerical_cols].corr()

# Buat visualisasi heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
# plt.show()

### SELESAI CODE ###

# Menampilkan histogram untuk semua kolom numerik (Opsional Skilled 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, column in enumerate(numerical_cols):
    ### MULAI CODE ###

    # Tampilkan histogram dan pastikan plot ditempatkan di subplot (axes) yang benar
    sns.histplot(df[column], bins=20, kde=True, color="skyblue", ax=axes[i])

    # Atur judul dan label
    axes[i].set_title(column)
    axes[i].set_xlabel("Nilai")
    axes[i].set_ylabel("Frekuensi")

    ### SELESAI CODE ###

plt.tight_layout()
# plt.show()

# Visualisasi yang lebih informatif (Opsional Advanced 1)

### MULAI CODE ###

plt.figure(figsize=(12, 6))

# Buat visualisasi boxplot untuk melihat sebaran 'TransactionAmount' (y) berdasarkan 'CustomerOccupation' (x)
sns.boxplot(x="CustomerOccupation", y="TransactionAmount", data=df)

plt.title("Nilai Transaksi per Pekerjaan Nasabah (Boxplot)")

# Putar label sumbu-x agar tidak tumpang tindih
plt.xticks(rotation=45)

# plt.show()

### SELESAI CODE ###

# -----------------------------------------------------------------
# (TANTANGAN OPSIONAL)
# -----------------------------------------------------------------
# Sekarang, bagaimana jika kita juga ingin melihat kepadatan distribusi data di setiap kategori?
# Coba buat visualisasi lain di bawah ini, misalnya 'violinplot' (sns.violinplot) dengan parameter yang sama.

# Mengecek dataset menggunakan isnull().sum()

### MULAI CODE ###
df.isnull().sum()

### SELESAI CODE ###
# Mengecek dataset menggunakan duplicated().sum()
# Melakukan feature scaling menggunakan MinMaxScaler() atau StandardScalar() untuk fitur numerik.
# Pastikan kamu menggunakan function head setelah melakukan scaling.
# Melakukan drop pada kolom yang memiliki keterangan id dan IP Address
# Melakukan feature encoding mengguanakan LabelEncoder() untuk fitur kategorikal.
# Pastikan kamu menggunakan function head setelah melakukan encoding.
# Last checking gunakan columns.tolist() untuk checking seluruh fitur yang ada.
# Perbaiki kode di bawah ini tanpa menambahkan atau mengurangi cell code ini.
# ___.columns.tolist()

# Menangani data yang hilang (bisa menggunakan dropna() atau metode imputasi fillna()).
# Menghapus data duplikat menggunakan drop_duplicates().

# Melakukan Handling Outlier Data berdasarkan jumlah outlier, apakah menggunakan metode drop atau mengisi nilai tersebut.
# Melakukan binning data berdasarkan kondisi rentang nilai pada fitur numerik,
# lakukan pada satu sampai dua fitur numerik.
# Silahkan lakukan encode hasil binning tersebut menggunakan LabelEncoder.
# Pastikan kamu mengerjakan tahapan ini pada satu cell.

# menyimpan model menggunakan joblib
# import joblib
# joblib.dump(___, "model_clustering.h5")

# Menghitung dan menampilkan nilai Silhouette Score.
# Membuat visualisasi hasil clustering

# Membangun model menggunakan PCA.
# ___ = PCA (n_components=<x>)
# ___ = ___.fit_transform(___)
# Menyimpan data PCA sebagai Dataframe dengan nama PCA_<numbers>
# <data_final> = pd.DataFrame(___, columns=['PCA1', 'PCA2', <sesuaikan dengan jumlah n>])
# Pastikan kamu membangun model Kmeans baru dengan data yang sudah dimodifikasi melalui PCA.
# ___ = KMeans(n_clusters=<x>)
# ___.fit(<data_final>)

# Simpan model PCA sebagai perbandingan dengan menjalankan cell code ini joblib.dump(model, "PC_model_clustering.h5")
# Pastikan yang disiimpan model yang sudah melalui .fit berdasarkan datasetyang sudah dilakukan PCA
# joblib.dump(___, "PCA_model_clustering.h5")

# Menampilkan analisis deskriptif minimal mean, min dan max untuk fitur numerik.
# Silakan menambahkan fungsi agregasi lainnya untuk experience lebih baik.
# pastikan output menghasilkan agregasi dan groupby bersamaan dengan mean, min, dan max

# Mengintegrasikan kembali data yang telah di-inverse dengan hasil cluster.
# Simpan Data
# ___.to_csv('data_clustering_incerse.csv', index=False)
