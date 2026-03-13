import sys
import os
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
# Setup Chinese font for Matplotlib
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign display
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QDateEdit, QLabel, QPushButton, QMessageBox, 
                             QFileDialog, QSplashScreen)
from PyQt6.QtGui import QPixmap, QFont, QColor
import pandas as pd
from PyQt6.QtCore import Qt, QDate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

from download_sst import fetch_himawari_sst_data, parse_himawari_sst

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class WeatherPlatform(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HIMAWARI 每日海面溫度監測平台")
        self.setGeometry(100, 100, 1200, 900)
        
        self.grid_data = None
        self.metadata = None
        self.show_isotherms = False # Toggle state for isotherms
        self.setup_ui()
        
        # Default to a recent available date (e.g. 2026-01-01 based on HIMAWARI crawler research)
        self.date_edit.setDate(QDate(2026, 1, 1))
        self.load_data()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top Bar: Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("選擇日期:"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        controls_layout.addWidget(self.date_edit)
        
        fetch_btn = QPushButton("獲取數據")
        fetch_btn.clicked.connect(self.load_data)
        controls_layout.addWidget(fetch_btn)

        self.export_btn = QPushButton("匯出 CSV")
        self.export_btn.clicked.connect(self.export_data)
        self.export_btn.setEnabled(False) # Enable only after data load
        controls_layout.addWidget(self.export_btn)

        self.isotherm_btn = QPushButton("顯示等溫線")
        self.isotherm_btn.setCheckable(True)
        self.isotherm_btn.clicked.connect(self.toggle_isotherms)
        controls_layout.addWidget(self.isotherm_btn)
        
        controls_layout.addStretch()
        
        self.status_label = QLabel("狀態: 準備就緒")
        controls_layout.addWidget(self.status_label)
        
        main_layout.addLayout(controls_layout)

        # Center: Visualization
        self.figure = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def toggle_isotherms(self, checked):
        self.show_isotherms = checked
        if checked:
            self.isotherm_btn.setText("隱藏等溫線")
        else:
            self.isotherm_btn.setText("顯示等溫線")
        self.update_plot()

    def load_data(self):
        qdate = self.date_edit.date()
        date = datetime(qdate.year(), qdate.month(), qdate.day())
        
        self.status_label.setText(f"狀態: 正在下載 {date.strftime('%Y-%m-%d')}...")
        QApplication.processEvents()
        
        raw_text = fetch_himawari_sst_data(date)
        if raw_text:
            self.metadata, self.grid_data = parse_himawari_sst(raw_text)
            if self.grid_data is not None:
                self.status_label.setText(f"狀態: 已載入 {date.strftime('%Y-%m-%d')}")
                self.export_btn.setEnabled(True)
                self.update_plot()
            else:
                self.status_label.setText("狀態: 解析失敗")
                QMessageBox.warning(self, "錯誤", "無法解析該日期的數據格式。")
        else:
            self.status_label.setText("狀態: 下載失敗")
            QMessageBox.warning(self, "錯誤", "無法從 JMA 下載該日期的數據。")

    def update_plot(self):
        if self.grid_data is None:
            return

        self.figure.clear()
        
        # Setup Projection
        ax = self.figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([118, 123, 21, 26], crs=ccrs.PlateCarree())
        
        # Data coordinates (0.1 degree resolution)
        lats = np.linspace(60.0, 0.0, 600)
        lons = np.linspace(100.0, 180.0, 800)
        
        # Higher resolution target: 0.05 degree
        # Visible extent: 118-123E, 21-26N. Add a bit buffer for interpolation.
        buffer = 0.5
        v_lons = np.arange(118 - buffer, 123 + buffer, 0.05)
        v_lats = np.arange(26 + buffer, 21 - buffer, -0.05)
        v_lon_grid, v_lat_grid = np.meshgrid(v_lons, v_lats)

        # 1. Slice data for interpolation area to save time
        lat_start_idx = int((60.0 - (26 + buffer)) * 10)
        lat_end_idx = int((60.0 - (21 - buffer)) * 10)
        lon_start_idx = int(((118 - buffer) - 100.0) * 10)
        lon_end_idx = int(((123 + buffer) - 100.0) * 10)
        
        lat_slice = slice(max(0, lat_start_idx), min(600, lat_end_idx))
        lon_slice = slice(max(0, lon_start_idx), min(800, lon_end_idx))
        
        sub_grid = self.grid_data[lat_slice, lon_slice]
        sub_lats = lats[lat_slice]
        sub_lons = lons[lon_slice]
        sub_lon_mesh, sub_lat_mesh = np.meshgrid(sub_lons, sub_lats)
        
        # 2. Prepare points for interpolation (filter out NaNs)
        valid_mask = ~np.isnan(sub_grid)
        points = np.stack([sub_lon_mesh[valid_mask], sub_lat_mesh[valid_mask]], axis=-1)
        values = sub_grid[valid_mask]
        
        if len(values) > 10: # Minimum points for interpolation
            # 3. Interpolate to 0.05 grid
            high_res_data = griddata(points, values, (v_lon_grid, v_lat_grid), method='linear')
            
            if not np.all(np.isnan(high_res_data)):
                vmin = np.nanmin(high_res_data)
                vmax = np.nanmax(high_res_data)
            else:
                vmin, vmax = 0, 35
                
            mesh = ax.pcolormesh(v_lon_grid, v_lat_grid, high_res_data, 
                               shading='auto', cmap='jet', transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax)
            
            # --- Draw Isotherms (Contours) if enabled ---
            if self.show_isotherms:
                # Calculate levels: e.g., every 1 degree
                if not np.all(np.isnan(high_res_data)):
                    d_min, d_max = np.nanmin(high_res_data), np.nanmax(high_res_data)
                    levels = np.arange(np.floor(d_min), np.ceil(d_max) + 1, 1.0)
                    if len(levels) > 1:
                        contours = ax.contour(v_lon_grid, v_lat_grid, high_res_data, 
                                            levels=levels, colors='white', alpha=0.6,
                                            linewidths=0.8, transform=ccrs.PlateCarree())
                        ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
            # --------------------------------------------
        else:
            # Fallback to original grid if not enough data for interpolation
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            vmin, vmax = 0, 35 # Default if local data empty
            mesh = ax.pcolormesh(lon_grid, lat_grid, self.grid_data, 
                               shading='auto', cmap='jet', transform=ccrs.PlateCarree(),
                               vmin=vmin, vmax=vmax)
        
        # Colorbar
        cbar = self.figure.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('溫度 (°C)')

        # Add Ultra-High Resolution Features (10m)
        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                              edgecolor='face',
                                              facecolor='#dddddd')
        coast_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                               edgecolor='black', facecolor='none')
        
        ax.add_feature(land_10m, zorder=2)
        ax.add_feature(coast_10m, linewidth=1.0, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', alpha=0.5, zorder=3)
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(f"HIMAWARI Daily SST - {self.metadata['date']}", fontsize=14)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def export_data(self):
        if self.grid_data is None:
            return
            
        date_str = self.date_edit.date().toString("yyyyMMdd")
        default_filename = f"him_sst_{date_str}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "匯出 CSV", default_filename, "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Convert 2D numpy array to DataFrame
                # Rows: 60N, 59.9N, ..., 0N
                # Cols: 100E, 100.1E, ..., 180E
                lats = np.linspace(60.0, 0.0, 600)
                lons = np.linspace(100.0, 180.0, 800)
                
                # Create DataFrame in grid format first
                df = pd.DataFrame(self.grid_data, index=lats, columns=lons)
                
                # Reshape to long format: [Latitude, Longitude, Temp]
                # In Pandas 3.0+, we dropna explicitly after stacking
                df_long = df.stack().dropna().reset_index()
                df_long.columns = ['緯度', '經度', '水溫']
                
                # Reorder to [經度, 緯度, 水溫] as requested
                df_long = df_long[['經度', '緯度', '水溫']]
                
                df_long.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"數據已匯出至: {file_path}\n(格式：經度、緯度、水溫)")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"匯出失敗: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # --- Splash Screen Setup ---
    # Attempt to load from bundled assets or current folder
    asset_path = resource_path("splash_bg.png")
    splash_pix = QPixmap(asset_path)
    splash = QSplashScreen(splash_pix)
    
    # Custom font for splash message
    splash_font = QFont("Microsoft YaHei", 12, QFont.Weight.Bold)
    splash.setFont(splash_font)
    splash.show()
    
    # Display loading message
    splash.showMessage("系統啟動中，正在載入地圖資源...", 
                      Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                      QColor("white"))
    app.processEvents()
    # ---------------------------

    window = WeatherPlatform()
    window.show()
    
    # Close splash screen once main window is ready
    splash.finish(window)
    
    sys.exit(app.exec())
