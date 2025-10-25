import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import sillhouette_score

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv"
df = pd.read_csv(url)

# Load data
# Tampilkan 5 baris pertamaa dengan fungsi head.
print(df.head())

# Tinjau jumlah baris kolom dan jenis data dalam dataset dengan info.
df.info()

# Menampilkan statistik deskriptif dataset dengan menjalankan describe

# Mengecek dataset menggunakan isnull().sum()
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
