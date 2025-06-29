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

# ============ CONFIG & STYLING ============
st.set_page_config(layout="wide")
st.title("📂 Composite Data Bor")
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============ FILE UPLOAD ============
uploaded_file = st.file_uploader("📄 Upload file Excel (.xlsx) (JANGAN ADA CONDITIONAL FORMATTING)", type=["xlsx"])
if not uploaded_file:
    st.info("Silakan upload file Excel yang berisi kolom: Prospect, Bukit, BHID, Layer, From, To, XCollar, YCollar, ZCollar, dan unsur.")
    st.stop()

df_raw = pd.read_excel(uploaded_file)
unsur = ['Ni','Co','Fe2O3','Fe','FeO','SiO2','CaO','MgO','MnO','Cr2O3','Al2O3','P2O5','TiO2','SO3','LOI','MC']
extra_cols = [col for col in ['Dens_WetMeas', 'Dens_WetArch'] if col in df_raw.columns]

if 'Thickness' not in df_raw.columns:
    df_raw['Thickness'] = df_raw['To'] - df_raw['From']

required = ['Prospect','Bukit','BHID','Layer','From','To','Thickness','XCollar','YCollar','ZCollar'] + unsur + extra_cols
missing = [c for c in required if c not in df_raw.columns]
for col in missing:
    df_raw[col] = np.nan

df_clean = df_raw[required].dropna(subset=['Prospect','Bukit','BHID','Layer','Thickness','XCollar','YCollar']).query("Thickness > 0")
sample_count = df_clean.groupby('BHID').size().reset_index(name='Sample_Count')

# ============ KOMPOSIT (CACHED) ============
@st.cache_data(show_spinner=False)
def create_composite(df_clean):
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
    return pd.DataFrame(result)

composite = create_composite(df_clean)
composite = composite.merge(df_clean.groupby('BHID')['To'].max().rename('Total_Depth'), on='BHID')
composite = composite.merge(sample_count, on='BHID', how='left')
composite['Percent'] = (composite['Layer Thickness'] / composite['Total_Depth']) * 100

# ============ KONVERSI KOORDINAT ============
transformer = Transformer.from_crs("EPSG:32751", "EPSG:4326", always_xy=True)
lonlat = composite.apply(lambda row: transformer.transform(row['XCollar'], row['YCollar']), axis=1)
composite['Longitude'] = lonlat.map(lambda x: x[0])
composite['Latitude'] = lonlat.map(lambda x: x[1])

# ============ SIDEBAR FILTER ============
st.sidebar.header("🔍 Filter Data")

df_filter_base = df_clean.copy()

prospect_opts = sorted(df_filter_base['Prospect'].unique())
selected_prospect = st.sidebar.selectbox("🏷️ Prospect", ["All"] + prospect_opts)
if selected_prospect != "All":
    df_filter_base = df_filter_base[df_filter_base['Prospect'] == selected_prospect]

bukit_opts = sorted(df_filter_base['Bukit'].unique())
selected_bukit = st.sidebar.multiselect("⛰️ Bukit", options=bukit_opts, default=bukit_opts)
df_filter_base = df_filter_base[df_filter_base['Bukit'].isin(selected_bukit)]

bhid_opts = sorted(df_filter_base['BHID'].unique())
selected_bhids = st.sidebar.multiselect("🔢 BHID", options=bhid_opts, default=bhid_opts)
df_filter_base = df_filter_base[df_filter_base['BHID'].isin(selected_bhids)]

layer_opts = sorted(df_filter_base['Layer'].astype(str).unique())
selected_layers = st.sidebar.multiselect("📚 Layer", options=layer_opts, default=layer_opts)
df_filter_base = df_filter_base[df_filter_base['Layer'].astype(str).isin(selected_layers)]

df_clean_filtered = df_filter_base.copy()

df_filter = composite[
    composite['BHID'].isin(df_clean_filtered['BHID']) &
    composite['Layer'].astype(str).isin(df_clean_filtered['Layer'].astype(str))
]

# ============ WARNA & LABEL LAYER ============
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

# ============ TAB ============
tab_data, tab_vis = st.tabs(["📍 Data & Peta", "📈 Visualisasi"])

with tab_data:
    st.markdown("## 📊 Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏷️ Prospect", df_filter['Prospect'].nunique())
    c2.metric("⛰️ Bukit", df_filter['Bukit'].nunique())
    c3.metric("🔢 BHID", df_filter['BHID'].nunique())
    c4.metric("🧪 Sampel Awal", df_clean_filtered.shape[0])

    st.markdown("## 🗘️ Peta Titik Bor")
    if not df_filter.empty:
        m = folium.Map(location=[df_filter['Latitude'].mean(), df_filter['Longitude'].mean()], zoom_start=12)
        for _, r in df_filter.iterrows():
            folium.CircleMarker(
                [r['Latitude'], r['Longitude']],
                radius=5, color='blue', fill=True, fill_opacity=0.7,
                popup=(f"Prospect: {r['Prospect']}<br>"
                       f"Bukit: {r['Bukit']}<br>"
                       f"BHID: {r['BHID']}<br>"
                       f"Layer: {r['Layer']}<br>"
                       f"Ni: {r['Ni']:.2f}")
            ).add_to(m)
       st_folium(m, height=650, use_container_width=True, returned_objects=[])

    st.markdown("### 📋 Tabel Data")
    show_original = st.checkbox("Tampilkan data asli (hanya mengunduh data komposit!)", value=False)
    composite_cols = ['Prospect','Bukit','BHID','Layer','From','To','Layer Thickness','Total_Depth'] + unsur
    original_cols = [col for col in composite_cols if col in df_clean.columns]
    if show_original:
        st.dataframe(df_clean_filtered[original_cols], use_container_width=True)
    else:
        st.dataframe(df_filter[composite_cols], use_container_width=True)

    st.markdown("### 📍 Koordinat Collar dan Total Depth")
    summary = df_filter[['Prospect','Bukit','BHID','XCollar','YCollar','ZCollar','Total_Depth']].drop_duplicates()
    st.dataframe(summary, use_container_width=True)

    st.markdown("### 📥 Unduh Hasil")
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df_filter.to_excel(writer, sheet_name='Composite', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
    st.download_button(
        label="⬇️ Download Excel (2 Sheet)",
        data=out.getvalue(),
        file_name="composite_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with tab_vis:
    st.markdown("### 📈 Visualisasi Komposit")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔺 Ternary Plot (SiO₂ - MgO - FeO)")
        ternary_data = df_clean_filtered.dropna(subset=['SiO2', 'MgO', 'FeO', 'Layer']).copy()
        ternary_data['Layer'] = ternary_data['Layer'].astype(int)
        ternary_data['Layer_Label'] = ternary_data['Layer'].map(layer_names)

        fig_tern = px.scatter_ternary(
            ternary_data,
            a='SiO2', b='MgO', c='FeO',
            color='Layer_Label',
            color_discrete_map={name: color_map[code] for code, name in layer_names.items()},
            hover_name='BHID',
            size_max=8
        )
        fig_tern.update_layout(height=500, margin=dict(t=40, b=40, l=20, r=20))
        st.plotly_chart(fig_tern, use_container_width=True)

    with col2:
        st.markdown("#### 📦 Box Plot MC per Layer")
        fig_box = go.Figure()
        for layer_code, layer_label in layer_names.items():
            df_layer = df_clean_filtered[df_clean_filtered['Layer'] == layer_code]
            if not df_layer.empty:
                fig_box.add_trace(go.Box(
                    y=df_layer['MC'],
                    name=f"{layer_code} - {layer_label}",
                    marker_color=color_map[layer_code],
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
            showlegend=False,
            margin=dict(t=40, b=40, l=20, r=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### ⚖️ Box Plot Densitas (Dens_WetMeas & Dens_WetArch)")
    fig_dens = go.Figure()
    densitas_types = {
        'Dens_WetMeas': 'Meas',
        'Dens_WetArch': 'Arch'
    }
    for dens_col, label in densitas_types.items():
        if dens_col not in df_clean_filtered.columns:
            continue
        for layer_code in [200, 300]:
            layer_data = df_clean_filtered[
                (df_clean_filtered['Layer'] == layer_code) &
                (df_clean_filtered[dens_col].notna())
            ]
            if not layer_data.empty:
                fig_dens.add_trace(go.Box(
                    y=layer_data[dens_col],
                    name=f"{layer_names[layer_code]} ({label})",
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

    st.markdown("#### 🔬 Scatter Plot (MgO vs Fe dan MgO vs SiO₂)")
    fig_scatter = px.scatter(
        df_clean_filtered.dropna(subset=['MgO', 'Fe', 'SiO2']),
        x='MgO', y='Fe', color=df_clean_filtered['Layer'].map(layer_names),
        labels={'color': 'Layer'},
        title='MgO vs Fe'
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    fig_scatter2 = px.scatter(
        df_clean_filtered.dropna(subset=['MgO', 'SiO2']),
        x='MgO', y='SiO2', color=df_clean_filtered['Layer'].map(layer_names),
        labels={'color': 'Layer'},
        title='MgO vs SiO₂'
    )
    fig_scatter2.update_layout(height=400)
    st.plotly_chart(fig_scatter2, use_container_width=True)
