import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ============ CONFIG ============
st.set_page_config(layout="wide")
st.title("\U0001F4C2 Composite Data Bor")

# ============ FILE UPLOAD MULTI ============
st.sidebar.markdown("### \U0001F4BE Riwayat Upload")
uploaded_files = st.sidebar.file_uploader("\U0001F4C4 Upload beberapa file Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Silakan upload satu atau lebih file Excel yang berisi kolom: Prospect, Bukit, BHID, Layer, From, To, XCollar, YCollar, ZCollar, dan unsur.")
    st.stop()

@st.cache_data
def load_multiple_files(file_objs):
    def standardize_column(col):
        col = col.strip().replace(" ", "")
        if len(col) == 1:
            return col.upper()
        return col[0].upper() + col[1:].lower()

    dfs_dict = {}
    for uploaded_file in file_objs:
        df = pd.read_excel(uploaded_file)
        df.columns = [standardize_column(col) for col in df.columns]
        df["Source_File"] = uploaded_file.name
        dfs_dict[uploaded_file.name] = df

    df_all = pd.concat(dfs_dict.values(), ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Bhid", "Xcollar", "Ycollar", "Zcollar", "Layer", "From", "To"])
    return dfs_dict, df_all

dfs_dict, raw_df = load_multiple_files(uploaded_files)

# ============ PILIH FILE ============
file_options = ["Gabungan (All Files)"] + list(dfs_dict.keys())
selected_file = st.sidebar.selectbox("\U0001F4C1 Pilih file untuk dilihat", file_options)

if selected_file != "Gabungan (All Files)":
    st.markdown(f"### \U0001F4C4 Menampilkan file: `{selected_file}`")
    st.dataframe(dfs_dict[selected_file], use_container_width=True)
    st.stop()

# ============ CLEANING & PREPARATION ============
unsur = ['Ni','Co','Fe2O3','Fe','FeO','SiO2','CaO','MgO','MnO','Cr2O3','Al2O3','P2O5','TiO2','SO3','LOI','MC']
unsur = [u[0].upper() + u[1:].lower() for u in unsur]
extra_cols = [col for col in raw_df.columns if col.lower() in ['dens_wetmeas', 'dens_wetarch']]

if 'Thickness' not in raw_df.columns:
    raw_df['Thickness'] = raw_df['To'] - raw_df['From']

required = ['Prospect','Bukit','Bhid','Layer','From','To','Thickness','Xcollar','Ycollar','Zcollar'] + unsur + extra_cols
for col in required:
    if col not in raw_df.columns:
        raw_df[col] = np.nan

df_clean = raw_df[required].dropna(subset=['Prospect','Bukit','Bhid','Layer','Thickness','Xcollar','Ycollar'])

@st.cache_data
def compute_composite(df_clean, unsur):
    result = []
    groups = list(df_clean.groupby(['Prospect','Bukit','Bhid','Layer']))
    for (prospect, bukit, bhid, layer), g in groups:
        row = {
            'Prospect': prospect,
            'Bukit': bukit,
            'Bhid': bhid,
            'Layer': layer,
            'From': g['From'].min(),
            'To': g['To'].max(),
            'Layer Thickness': g['Thickness'].sum(),
            'Xcollar': g['Xcollar'].iat[0],
            'Ycollar': g['Ycollar'].iat[0],
            'Zcollar': g['Zcollar'].iat[0]
        }
        for u in unsur:
            row[u] = np.average(g[u], weights=g['Thickness']) if g[u].notna().any() else np.nan
        result.append(row)
    composite = pd.DataFrame(result)
    composite = composite.merge(df_clean.groupby('Bhid')['To'].max().rename('Total_Depth'), on='Bhid')
    composite['Percent'] = (composite['Layer Thickness'] / composite['Total_Depth']) * 100
    return composite

@st.cache_data
def transform_coordinates(composite):
    transformer = Transformer.from_crs("EPSG:32751", "EPSG:4326", always_xy=True)
    x = composite['Xcollar'].to_numpy()
    y = composite['Ycollar'].to_numpy()
    lon, lat = transformer.transform(x, y)
    composite['Longitude'] = lon
    composite['Latitude'] = lat
    return composite

with st.spinner("\u23F3 Menghitung komposit dan koordinat..."):
    composite = transform_coordinates(compute_composite(df_clean, unsur))

# ============ FILTER ============
st.sidebar.header("\U0001F50D Filter Data")
filter_base = df_clean.copy()
prospect_opts = sorted(filter_base['Prospect'].dropna().unique())
selected_prospect = st.sidebar.selectbox("\U0001F3F7️ Prospect", ["All"] + prospect_opts)
if selected_prospect != "All":
    filter_base = filter_base[filter_base['Prospect'] == selected_prospect]

bukit_opts = sorted(filter_base['Bukit'].dropna().unique())
selected_bukits = st.sidebar.multiselect("\u26F0\uFE0F Bukit", options=bukit_opts, default=bukit_opts)
filter_base = filter_base[filter_base['Bukit'].isin(selected_bukits)]

bhid_opts = sorted(filter_base['Bhid'].unique())
selected_bhids = st.sidebar.multiselect("\U0001F522 BHID", options=bhid_opts, default=bhid_opts)
filter_base = filter_base[filter_base['Bhid'].isin(selected_bhids)]

layer_opts = sorted(filter_base['Layer'].astype(str).unique())
selected_layers = st.sidebar.multiselect("\U0001F4DA Layer", options=layer_opts, default=layer_opts)
filter_base = filter_base[filter_base['Layer'].astype(str).isin(selected_layers)]

filtered_composite = composite[
    composite['Bhid'].isin(filter_base['Bhid']) &
    composite['Layer'].astype(str).isin(filter_base['Layer'].astype(str))
]

# ============ RINGKASAN & PETA =========
st.markdown("## \U0001F4CA Ringkasan")
if not filtered_composite.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prospect", filtered_composite['Prospect'].nunique())
    c2.metric("Bukit", filtered_composite['Bukit'].nunique())
    c3.metric("BHID", filtered_composite['Bhid'].nunique())
    c4.metric("Sampel", filter_base.shape[0])
else:
    st.warning("Tidak ada data yang sesuai dengan filter.")

st.markdown("## \U0001F6B6‍ Peta Bor")
if not filtered_composite.empty:
    m = folium.Map(location=[filtered_composite['Latitude'].mean(), filtered_composite['Longitude'].mean()], zoom_start=12)
    for _, r in filtered_composite.iterrows():
        folium.CircleMarker(
            [r['Latitude'], r['Longitude']],
            radius=5, color='blue', fill=True, fill_opacity=0.7,
            popup=f"BHID: {r['Bhid']}<br>Layer: {r['Layer']}"
        ).add_to(m)
    st_folium(m, height=500, use_container_width=True)
