import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import io

# Import our custom downloader logic
from download_sst import fetch_himawari_sst_data, parse_himawari_sst

# Set page config
st.set_page_config(page_title="HIMAWARI 海面溫度監測平台", layout="wide")

st.title("🌊 HIMAWARI 每日海面溫度監測平台")
st.markdown("這是一個基於 JMA HIMAWARI 衛星數據的即時海面溫度 (SST) 監測工具。")

# Sidebar for controls
st.sidebar.header("控制面板")

# Date Selection
selected_date = st.sidebar.date_input("選擇日期", value=datetime(2026, 1, 1))
fetch_btn = st.sidebar.button("獲取並更新數據")

# Options
show_isotherms = st.sidebar.checkbox("顯示等溫線", value=False)

# Initialize Session State
if 'grid_data' not in st.session_state:
    st.session_state.grid_data = None
if 'last_date' not in st.session_state:
    st.session_state.last_date = None

def load_data(date):
    with st.spinner(f"正在從 JMA 下載 {date.strftime('%Y-%m-%d')} 的數據..."):
        raw_text = fetch_himawari_sst_data(date)
        if raw_text:
            metadata, grid_data = parse_himawari_sst(raw_text)
            if grid_data is not None:
                st.session_state.grid_data = grid_data
                st.session_state.last_date = date
                return True
            else:
                st.error("解析失敗：無法解析該日期的數據格式。")
        else:
            st.error("下載失敗：無法從 JMA 獲取數據。")
    return False

if fetch_btn:
    load_data(selected_date)

# Main Visualization
if st.session_state.grid_data is not None:
    grid_data = st.session_state.grid_data
    
    st.info(f"當前顯示日期: {st.session_state.last_date.strftime('%Y-%m-%d')}")
    
    # Visualization logic (Same as desktop)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([118, 123, 21, 26], crs=ccrs.PlateCarree())
    
    lats = np.linspace(60.0, 0.0, 600)
    lons = np.linspace(100.0, 180.0, 800)
    
    # 0.05 Resolution Interpolation
    buffer = 0.5
    v_lons = np.arange(118 - buffer, 123 + buffer, 0.05)
    v_lats = np.arange(26 + buffer, 21 - buffer, -0.05)
    v_lon_grid, v_lat_grid = np.meshgrid(v_lons, v_lats)

    lat_start_idx = int((60.0 - (26 + buffer)) * 10)
    lat_end_idx = int((60.0 - (21 - buffer)) * 10)
    lon_start_idx = int(((118 - buffer) - 100.0) * 10)
    lon_end_idx = int(((123 + buffer) - 100.0) * 10)
    
    lat_slice = slice(max(0, lat_start_idx), min(600, lat_end_idx))
    lon_slice = slice(max(0, lon_start_idx), min(800, lon_end_idx))
    
    sub_grid = grid_data[lat_slice, lon_slice]
    sub_lats = lats[lat_slice]
    sub_lons = lons[lon_slice]
    sub_lon_mesh, sub_lat_mesh = np.meshgrid(sub_lons, sub_lats)
    
    valid_mask = ~np.isnan(sub_grid)
    points = np.stack([sub_lon_mesh[valid_mask], sub_lat_mesh[valid_mask]], axis=-1)
    values = sub_grid[valid_mask]
    
    if len(values) > 10:
        high_res_data = griddata(points, values, (v_lon_grid, v_lat_grid), method='linear')
        vmin = np.nanmin(high_res_data) if not np.all(np.isnan(high_res_data)) else 0
        vmax = np.nanmax(high_res_data) if not np.all(np.isnan(high_res_data)) else 35
        
        mesh = ax.pcolormesh(v_lon_grid, v_lat_grid, high_res_data, 
                           shading='auto', cmap='jet', transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=vmax)
        
        if show_isotherms and not np.all(np.isnan(high_res_data)):
            d_min, d_max = np.nanmin(high_res_data), np.nanmax(high_res_data)
            levels = np.arange(np.floor(d_min), np.ceil(d_max) + 1, 1.0)
            if len(levels) > 1:
                contours = ax.contour(v_lon_grid, v_lat_grid, high_res_data, 
                                    levels=levels, colors='white', alpha=0.6,
                                    linewidths=0.8, transform=ccrs.PlateCarree())
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
    else:
        # Fallback
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        mesh = ax.pcolormesh(lon_grid, lat_grid, grid_data, 
                           shading='auto', cmap='jet', transform=ccrs.PlateCarree(),
                           vmin=0, vmax=35)

    plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8, label='溫度 (°C)')
    
    # Features
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='#dddddd')
    coast_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none')
    ax.add_feature(land_10m, zorder=2)
    ax.add_feature(coast_10m, linewidth=1.0, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', alpha=0.5, zorder=3)
    
    ax.set_title(f"HIMAWARI SST ({st.session_state.last_date.strftime('%Y-%m-%d')})", fontsize=14)
    
    # Show in Streamlit
    st.pyplot(fig)

    # Export CSV Support
    st.sidebar.markdown("---")
    st.sidebar.subheader("數據導出")
    
    # Re-calculate long format for CSV
    df = pd.DataFrame(grid_data, index=lats, columns=lons)
    df_long = df.stack().dropna().reset_index()
    df_long.columns = ['緯度', '經度', '水溫']
    df_long = df_long[['經度', '緯度', '水溫']]
    
    csv = df_long.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.sidebar.download_button(
        label="下載當前數據 (CSV)",
        data=csv,
        file_name=f"sst_data_{st.session_state.last_date.strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
else:
    st.warning("請先在側邊欄選擇日期並點擊「獲取並更新數據」。")
