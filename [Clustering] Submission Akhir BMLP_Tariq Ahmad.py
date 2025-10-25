import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import sillhouette_score

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv"
df = pd.read_csv(url)

# Load data
# Tampilkan 5 baris pertamaa dengan fungsi head.
df.head()

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
#
# Menangani data yang hilang (bisa menggunakan dropna() atau metode imputasi fillna()).
# Menghapus data duplikat menggunakan drop_duplicates().
#
# Melakukan Handling Outlier Data berdasarkan jumlah outlier, apakah menggunakan metode drop atau mengisi nilai tersebut.
# Melakukan binning data berdasarkan kondisi rentang nilai pada fitur numerik,
# lakukan pada satu sampai dua fitur numerik.
# Silahkan lakukan encode hasil binning tersebut menggunakan LabelEncoder.
# Pastikan kamu mengerjakan tahapan ini pada satu cell.
