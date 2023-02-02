import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
from PIL import Image

# dataset = pd.read_csv('data_final.csv') # For what I have in mind I'll need the data
# model = pickle.load(open('pipeline.pkl', 'rb')) # We load the trained model

st.title(""" Web App Clustering Data Penjualan dengan K-Means """)
# st.write(""" ##### Aplikasi berbasis WEB untuk melakukan Clustering Data Penjualan """)

with st.sidebar:
    selected = option_menu("Main Menu", ["Visualisasi", 'Masukkan Jumlah Barang'], 
        icons=['gear', 'gear'], menu_icon="cast", default_index=1)
    selected
    

img = Image.open('toko.jpg')
img = img.resize((680,250))
st.image(img, use_column_width=False)

if (selected == 'Masukkan Jumlah Barang') :
    
    jumlah_item = st.slider("Masukkan Jumlah Barang", 0, 6000)
    klik = st.button('Silahkan Cek')
    
    if klik :
        if jumlah_item <= 350 :
            st.write('Cluster 0')
        elif jumlah_item >= 400 and jumlah_item <= 1200 :
            st.write('Cluster 1')
        elif jumlah_item >= 4000 and jumlah_item <= 6000 :
            st.write('Cluster 2')
        elif jumlah_item >= 1500 and jumlah_item <= 3000 :
            st.write('Cluster 3')
        else:
            st.write('Not Cluster')
            
    st.write(" ##### Cluster 0   : \n",
             " ##### Cluster 1   : \n",
             " ##### Cluster 2   : \n",
             " ##### Cluster 3   : \n",
             " ##### Not Cluster : \n")

if (selected == 'Visualisasi') :
    
#     id_item = st.number_input("Masukkan Kode", 0, 999)
#     klik = st.button('Silahkan Cek')
    
    sns.set_theme()
    # -----------------------------------------------------------

    # Helper functions
    # -----------------------------------------------------------
    # Load data from external source
    @st.cache
    def load_data():
        df = pd.read_csv('data_clustering.csv')
        return df
    df = load_data()

    # Load data from external source

    def run_kmeans(df, n_clusters=4):
        kmeans = KMeans(n_clusters, random_state=0).fit(df[["jumlah_item", "satuan_brg"]])

        fig, ax = plt.subplots(figsize=(16, 9))

        #Create scatterplot
        ax = sns.scatterplot(
            ax=ax,
            x=df.jumlah_item,
            y=df.satuan_brg,
            hue=kmeans.labels_,
            palette=sns.color_palette("colorblind", n_colors=n_clusters),
            legend=None,
        )

        return fig

    # Sidebar
    # -----------------------------------------------------------
    sidebar = st.sidebar
    df_display = sidebar.checkbox("Display Raw Data", value=True)

    n_clusters = sidebar.slider(
        "Pilih Jumlah Cluster",
        min_value=1,
        max_value=4,
    )

    # Main
    # -----------------------------------------------------------
    # Create a title for your app
    st.title("K-Means Clustering")

    # A description
    st.write("Berikut adalah visualisasi dataset yang digunakan dalam analisis ini:")

    # Show cluster scatter plot
    st.write(run_kmeans(df, n_clusters=n_clusters))

    # Display the dataframe
    #st.write(df)

    # Display the dataframe
    #df_display = st.checkbox("Display Raw Data", value=True)

    if df_display:
        st.write(df)
    # -----------------------------------------------------------

# st.sidebar.header('Upload your Excel file')
# upload_file = st.sidebar.file_uploader('')
# if upload_file is not None:
#     inputan = pd.read_csv(upload_file)
# else:
#     def input_user():
#         st.sidebar.text('')
#         st.sidebar.header('INPUTAN USER')
        
#         total = st.sidebar.slider('Masukkan Jumlah Item', 0, 999)
#         satuan = st.sidebar.slider('Masukkan Satuan Barang', 0, 12)
        
#         data = {'jumlah_item': total,
#                 'satuan_brg' : satuan}
        
#         fitur = pd.DataFrame(data, index=[0])
#         return fitur
    
#     inputan = input_user()
    
# # Menggabungkan inputan dan dataset
# dataku = pd.read_csv("hasil_clustering.csv")
# penjualan = dataku.drop(columns=['id','cluster'])
# df = pd.concat([inputan, penjualan], axis=0)

# # Menampilkan parameter hasil inputan
# st.subheader('Parameter Inputan')

# if upload_file is not None:
#     st.write(df)
# else:
#     st.write('Waiting for the csv file to upload..')
#     st.write(df)
    
# # Load save model
# load_model = pickle.load(open('pipeline2.pkl', 'rb'))

# # Terapkan Kmeans
# clstr = load_model.predict(df)

# st.subheader('Keterangan Cluster')
# cluster = np.array(['0', '1', '2', '3'])
# st.write(cluster)

# st.subheader('Hasil Clustering Data Penjualan')
# st.write(cluster[clstr])
            
