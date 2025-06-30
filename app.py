import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import threading
import warnings
import os
import tempfile
import uuid

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

@@ -25,15 +27,29 @@
""", unsafe_allow_html=True)

# ============ FILE UPLOAD ============
st.sidebar.markdown("### \U0001F4BE Riwayat Upload")
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

uploaded_file = st.file_uploader("\U0001F4C4 Upload file Excel (.xlsx) (JANGAN ADA CONDITIONAL FORMATTING)", type=["xlsx"])
if not uploaded_file:
    st.info("Silakan upload file Excel yang berisi kolom: Prospect, Bukit, BHID, Layer, From, To, XCollar, YCollar, ZCollar, dan unsur.")
if uploaded_file:
    unique_id = str(uuid.uuid4())[:8]
    temp_path = os.path.join(tempfile.gettempdir(), f"{unique_id}.xlsx")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.session_state.uploaded_files[uploaded_file.name] = temp_path

if not st.session_state.uploaded_files:
    st.info("Silakan upload minimal satu file Excel.")
st.stop()

selected_filename = st.sidebar.selectbox("Pilih file yang ingin dianalisis:", list(st.session_state.uploaded_files.keys()))
selected_filepath = st.session_state.uploaded_files[selected_filename]

# ============ CACHING FUNCTIONS ============
@st.cache_data
def load_and_prepare_data(file):
    df_raw = pd.read_excel(file)
def load_and_prepare_data(filepath):
    df_raw = pd.read_excel(filepath)
unsur = ['Ni','Co','Fe2O3','Fe','FeO','SiO2','CaO','MgO','MnO','Cr2O3','Al2O3','P2O5','TiO2','SO3','LOI','MC']
extra_cols = [col for col in ['Dens_WetMeas', 'Dens_WetArch'] if col in df_raw.columns]

@@ -79,85 +95,67 @@
@st.cache_data
def transform_coordinates(composite):
transformer = Transformer.from_crs("EPSG:32751", "EPSG:4326", always_xy=True)
    lonlat = composite.apply(lambda row: transformer.transform(row['XCollar'], row['YCollar']), axis=1)
    composite['Longitude'] = lonlat.map(lambda x: x[0])
    composite['Latitude'] = lonlat.map(lambda x: x[1])
    x = composite['XCollar'].to_numpy()
    y = composite['YCollar'].to_numpy()
    lon, lat = transformer.transform(x, y)
    composite['Longitude'] = lon
    composite['Latitude'] = lat
return composite

# ============ BACKGROUND PROCESSING ============
def run_heavy_processing(df_clean, unsur, result):
    comp = compute_composite(df_clean, unsur)
    comp = transform_coordinates(comp)
    result['composite'] = comp

if 'composite_ready' not in st.session_state:
    df_clean, unsur, extra_cols = load_and_prepare_data(uploaded_file)
    result = {}
    thread = threading.Thread(target=run_heavy_processing, args=(df_clean, unsur, result))
    thread.start()
    with st.spinner("\u23F3 Menghitung komposit dan koordinat..."):
        thread.join()
    st.session_state['df_clean'] = df_clean
    st.session_state['unsur'] = unsur
    st.session_state['extra_cols'] = extra_cols
    st.session_state['composite'] = result['composite']
    st.session_state['composite_ready'] = True

# Retrieve session state
composite = st.session_state['composite']
df_clean = st.session_state['df_clean']
unsur = st.session_state['unsur']
extra_cols = st.session_state['extra_cols']
# ============ DATA PROCESSING ============
with st.spinner("\u23F3 Menghitung komposit dan koordinat..."):
    df_clean, unsur, extra_cols = load_and_prepare_data(selected_filepath)
    composite = transform_coordinates(compute_composite(df_clean, unsur))

# ============ FILTERS ============
st.sidebar.header("\U0001F50D Filter Data")

filter_base = df_clean.copy()
prospect_opts = sorted(filter_base['Prospect'].unique())
selected_prospect = st.sidebar.selectbox("\U0001F3F7️ Prospect", ["All"] + prospect_opts)
if selected_prospect != "All":
filter_base = filter_base[filter_base['Prospect'] == selected_prospect]

bukit_opts = sorted(filter_base['Bukit'].unique())
selected_bukits = st.sidebar.multiselect("\u26F0\uFE0F Bukit", options=bukit_opts, default=bukit_opts)
filter_base = filter_base[filter_base['Bukit'].isin(selected_bukits)]

bhid_opts = sorted(filter_base['BHID'].unique())
selected_bhids = st.sidebar.multiselect("\U0001F522 BHID", options=bhid_opts, default=bhid_opts)
filter_base = filter_base[filter_base['BHID'].isin(selected_bhids)]

layer_opts = sorted(filter_base['Layer'].astype(str).unique())
selected_layers = st.sidebar.multiselect("\U0001F4DA Layer", options=layer_opts, default=layer_opts)
filter_base = filter_base[filter_base['Layer'].astype(str).isin(selected_layers)]

filtered_composite = composite[
composite['BHID'].isin(filter_base['BHID']) &
composite['Layer'].astype(str).isin(filter_base['Layer'].astype(str))
]

# ============ PLOTTING ============
st.markdown("## \U0001F4CA Ringkasan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prospect", filtered_composite['Prospect'].nunique())
c2.metric("Bukit", filtered_composite['Bukit'].nunique())
c3.metric("BHID", filtered_composite['BHID'].nunique())
c4.metric("Sampel", filter_base.shape[0])

st.markdown("## \U0001F6B6\u200D Peta Bor")
if not filtered_composite.empty:
m = folium.Map(location=[filtered_composite['Latitude'].mean(), filtered_composite['Longitude'].mean()], zoom_start=12)
for _, r in filtered_composite.iterrows():
folium.CircleMarker(
[r['Latitude'], r['Longitude']],
radius=5, color='blue', fill=True, fill_opacity=0.7,
popup=f"BHID: {r['BHID']}<br>Layer: {r['Layer']}"
).add_to(m)
with st.container():
st_folium(m, height=500, use_container_width=True)

# ============ EXPORT ============
st.markdown("### \U0001F4E5 Download")
out = BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
filtered_composite.to_excel(writer, index=False, sheet_name="Composite")
st.download_button("\u2B07\uFE0F Unduh Hasil", data=out.getvalue(), file_name="composite_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
