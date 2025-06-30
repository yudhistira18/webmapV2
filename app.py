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

# ============ FILE UPLOAD MULTI ============
st.sidebar.markdown("### \U0001F4BE Riwayat Upload")
uploaded_files = st.sidebar.file_uploader("\U0001F4C4 Upload beberapa file Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Silakan upload satu atau lebih file Excel yang berisi kolom: Prospect, Bukit, BHID, Layer, From, To, XCollar, YCollar, ZCollar, dan unsur.")
    st.stop()

@st.cache_data
def load_multiple_files(file_objs):
    dfs = []
    for uploaded_file in file_objs:
        df = pd.read_excel(uploaded_file)
        df["Source_File"] = uploaded_file.name
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["BHID", "XCollar", "YCollar", "ZCollar", "Layer", "From", "To"])
    return df_all

raw_df = load_multiple_files(uploaded_files)

# ============ CLEANING & PREPARATION ============
unsur = ['Ni','Co','Fe2O3','Fe','FeO','SiO2','CaO','MgO','MnO','Cr2O3','Al2O3','P2O5','TiO2','SO3','LOI','MC']
extra_cols = [col for col in ['Dens_WetMeas', 'Dens_WetArch'] if col in raw_df.columns]

if 'Thickness' not in raw_df.columns:
    raw_df['Thickness'] = raw_df['To'] - raw_df['From']

required = ['Prospect','Bukit','BHID','Layer','From','To','Thickness','XCollar','YCollar','ZCollar'] + unsur + extra_cols
for col in required:
    if col not in raw_df.columns:
        raw_df[col] = np.nan

df_clean = raw_df[required].dropna(subset=['Prospect','Bukit','BHID','Layer','Thickness','XCollar','YCollar'])

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

with st.spinner("\u23F3 Menghitung komposit dan koordinat..."):
    composite = transform_coordinates(compute_composite(df_clean, unsur))

# ============ FILTER ============
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

# ============ PETA & TABEL ============
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

# ============ TABEL DATA ============
st.markdown("## \U0001F4CB Tabel Data")
show_original = st.checkbox("Tampilkan data asli (hanya mengunduh data komposit!)", value=False)
composite_cols = ['Prospect','Bukit','BHID','Layer','From','To','Layer Thickness','Total_Depth'] + unsur
original_cols = [col for col in composite_cols if col in df_clean.columns]

if show_original:
    st.dataframe(filter_base[original_cols], use_container_width=True)
else:
    st.dataframe(filtered_composite[composite_cols], use_container_width=True)

st.markdown("### \U0001F4CA Koordinat Collar dan Total Depth")
summary = filtered_composite[['Prospect','Bukit','BHID','XCollar','YCollar','ZCollar','Total_Depth']].drop_duplicates()
st.dataframe(summary, use_container_width=True)

# ============ VISUALISASI ============
st.markdown("## 📈 Visualisasi")
tab1, tab2 = st.tabs(["🔺 Ternary & Boxplot", "📊 Scatter MgO"])

with tab1:
    st.markdown("### 🔺 Ternary Plot (SiO₂ - MgO - FeO)")
    ternary_data = filter_base.dropna(subset=['SiO2', 'MgO', 'FeO', 'Layer']).copy()
    ternary_data['Layer'] = ternary_data['Layer'].astype(int)
    layer_names = {
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
    ternary_data['Layer_Label'] = ternary_data['Layer'].map(layer_names)

    fig_tern = px.scatter_ternary(
        ternary_data,
        a='SiO2', b='MgO', c='FeO',
        color='Layer_Label',
        color_discrete_map={name: color_map[code] for code, name in layer_names.items() if code in color_map},
        hover_name='BHID'
    )
    fig_tern.update_layout(height=500, margin=dict(t=40, b=40, l=20, r=20))
    st.plotly_chart(fig_tern, use_container_width=True)

    st.markdown("### 📦 Boxplot MC per Layer")
    fig_box = go.Figure()
    for code, label in layer_names.items():
        df_layer = filter_base[filter_base['Layer'] == code]
        if not df_layer.empty and 'MC' in df_layer:
            fig_box.add_trace(go.Box(
                y=df_layer['MC'],
                name=f"{code} - {label}",
                marker_color=color_map.get(code, 'gray'),
                boxpoints='all',
                jitter=0.4,
                pointpos=0,
                marker=dict(opacity=0.6, size=4),
                line=dict(width=1)
            ))
    fig_box.update_layout(
        yaxis_title="MC (%)",
        xaxis_title="Layer",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### ⚖️ Boxplot Densitas")
    fig_dens = go.Figure()
    for dens_col, label in {'Dens_WetMeas': 'Meas', 'Dens_WetArch': 'Arch'}.items():
        if dens_col in filter_base.columns:
            for code in [200, 300]:
                df_dens = filter_base[(filter_base['Layer'] == code) & filter_base[dens_col].notna()]
                if not df_dens.empty:
                    fig_dens.add_trace(go.Box(
                        y=df_dens[dens_col],
                        name=f"{layer_names.get(code, code)} ({label})",
                        marker_color=color_map.get(code, 'gray'),
                        boxpoints='all',
                        jitter=0.4,
                        pointpos=0
                    ))
    fig_dens.update_layout(
        yaxis_title="Densitas (gr/cm³)",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_dens, use_container_width=True)

with tab2:
    st.markdown("### 🔬 Scatter Plot MgO vs Fe")
    fig1 = px.scatter(
        filter_base.dropna(subset=['MgO','Fe']),
        x='MgO', y='Fe', color=filter_base['Layer'].map(layer_names),
        labels={'color': 'Layer'}, title='MgO vs Fe'
    )
    fig1.update_layout(height=450)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🔬 Scatter Plot MgO vs SiO₂")
    fig2 = px.scatter(
        filter_base.dropna(subset=['MgO','SiO2']),
        x='MgO', y='SiO2', color=filter_base['Layer'].map(layer_names),
        labels={'color': 'Layer'}, title='MgO vs SiO₂'
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)


# ============ EXPORT ============
st.markdown("### \U0001F4E5 Download")
out = BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    filtered_composite.to_excel(writer, index=False, sheet_name="Composite")
    summary.to_excel(writer, index=False, sheet_name="Summary")
st.download_button("\u2B07\uFE0F Unduh Hasil", data=out.getvalue(), file_name="composite_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

punya saya, ingat
