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
