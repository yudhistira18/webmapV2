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

# ============ CONFIG & STYLING ============
st.set_page_config(layout="wide")
st.title("\U0001F4C2 Composite Data Bor")
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
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
    st.stop()

selected_filename = st.sidebar.selectbox("Pilih file yang ingin dianalisis:", list(st.session_state.uploaded_files.keys()))
selected_filepath = st.session_state.uploaded_files[selected_filename]

# ============ CACHING FUNCTIONS ============
@st.cache_data
def load_and_prepare_data(filepath):
    df_raw = pd.read_excel(filepath)
    unsur = ['Ni','Co','Fe2O3','Fe','FeO','SiO2','CaO','MgO','MnO','Cr2O3','Al2O3','P2O5','TiO2','SO3','LOI','MC']
    extra_cols = [col for col in ['Dens_WetMeas', 'Dens_WetArch'] if col in df_raw.columns]

    if 'Thickness' not in df_raw.columns:
        df_raw['Thickness'] = df_raw['To'] - df_raw['From']

    required = ['Prospect','Bukit','BHID','Layer','From','To','Thickness','XCollar','YCollar','ZCollar'] + unsur + extra_cols
    for col in required:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    df_clean = df_raw[required].dropna(subset=['Prospect','Bukit','BHID','Layer','Thickness','XCollar','YCollar'])
    return df_clean, unsur, extra_cols

@st.cache_data
def compute_composite(df_clean, unsur):
    result = []
    groups = list(df_clean.groupby(['Prospect','Bukit','BHID','Layer']))
    for (prospect, bukit, bhid, layer), g in groups:
        row = {
            'Prospect': prospect,
            'Bukit': bukit,
            'BHID': bhid,
            'Layer': layer,
            'From': g['From'].min(),
            'To': g['To'].max(),
            'Layer Thickness': g['Thickness'].sum(),
            'XCollar': g['XCollar'].iat[0],
            'YCollar': g['YCollar'].iat[0],
            'ZCollar': g['ZCollar'].iat[0]
        }
        for u in unsur:
            row[u] = np.average(g[u], weights=g['Thickness']) if g[u].notna().any() else np.nan
        result.append(row)
    composite = pd.DataFrame(result)
    composite = composite.merge(df_clean.groupby('BHID')['To'].max().rename('Total_Depth'), on='BHID')
    composite['Percent'] = (composite['Layer Thickness'] / composite['Total_Depth']) * 100
    return composite

@st.cache_data
def transform_coordinates(composite):
    transformer = Transformer.from_crs("EPSG:32751", "EPSG:4326", always_xy=True)
    x = composite['XCollar'].to_numpy()
    y = composite['YCollar'].to_numpy()
    lon, lat = transformer.transform(x, y)
    composite['Longitude'] = lon
    composite['Latitude'] = lat
    return composite

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
        st_folium(m, height=500, use_container_width=True, returned_objects=[])

# ============ VISUALIZATION TAB ============
st.markdown("## \U0001F4C8 Visualisasi")

layer_map = {
    100: 'Top Soil',
    200: 'Limonit',
    250: 'Limonit Organik',
    300: 'Saprolit',
    400: 'Bedrock'
}
color_map = {
    100: 'gray',
    200: 'red',
    250: 'black',
    300: 'green',
    400: 'blue'
}

with st.expander("\U0001F52C Scatter Plot (MgO vs Fe dan MgO vs SiO₂)"):
    scatter_data = filter_base.dropna(subset=['MgO', 'Fe', 'SiO2', 'Layer']).copy()
    scatter_data['Layer_Label'] = scatter_data['Layer'].map(layer_map)

    fig_scatter1 = px.scatter(
        scatter_data, x='MgO', y='Fe', color='Layer_Label',
        labels={'color': 'Layer'}, title='MgO vs Fe'
    )
    st.plotly_chart(fig_scatter1, use_container_width=True)

    fig_scatter2 = px.scatter(
        scatter_data, x='MgO', y='SiO2', color='Layer_Label',
        labels={'color': 'Layer'}, title='MgO vs SiO₂'
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)

with st.expander("\u26AA Ternary Plot (SiO₂ - MgO - FeO)"):
    ternary_data = filter_base.dropna(subset=['SiO2', 'MgO', 'FeO', 'Layer']).copy()
    ternary_data['Layer_Label'] = ternary_data['Layer'].map(layer_map)

    fig_tern = px.scatter_ternary(
        ternary_data, a='SiO2', b='MgO', c='FeO',
        color='Layer_Label',
        color_discrete_map={name: color_map[code] for code, name in layer_map.items()},
        hover_name='BHID',
        size_max=8
    )
    fig_tern.update_layout(height=500, margin=dict(t=40, b=40, l=20, r=20))
    st.plotly_chart(fig_tern, use_container_width=True)

with st.expander("\U0001F4A1 Box Plot Densitas"):
    fig_dens = go.Figure()
    densitas_types = {
        'Dens_WetMeas': 'Meas',
        'Dens_WetArch': 'Arch'
    }
    for dens_col, label in densitas_types.items():
        if dens_col not in filter_base.columns:
            continue
        for layer_code in [200, 300]:
            layer_data = filter_base[
                (filter_base['Layer'] == layer_code) &
                (filter_base[dens_col].notna())
            ]
            if not layer_data.empty:
                fig_dens.add_trace(go.Box(
                    y=layer_data[dens_col],
                    name=f"{layer_map[layer_code]} ({label})",
                    marker_color=color_map[layer_code],
                    boxpoints='all',
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(opacity=0.6, size=4),
                    line=dict(width=1)
                ))
    fig_dens.update_layout(
        yaxis_title="Densitas (gr/cm³)",
        xaxis_title="Layer & Jenis Densitas",
        height=500,
        showlegend=False,
        margin=dict(t=40, b=40, l=20, r=20)
    )
    st.plotly_chart(fig_dens, use_container_width=True)

# ============ EXPORT ============
st.markdown("### \U0001F4E5 Download")
out = BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    filtered_composite.to_excel(writer, index=False, sheet_name="Composite")
st.download_button("\u2B07\uFE0F Unduh Hasil", data=out.getvalue(), file_name="composite_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
