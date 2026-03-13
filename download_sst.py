import requests  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime

def fetch_himawari_sst_data(date: datetime):
    """
    Fetches the HIMAWARI daily SST data for a specific date.
    URL Pattern: https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/YYYY/him_sst_pac_DYYYYMMDD.txt
    """
    year = date.strftime("%Y")
    date_str = date.strftime("%Y%m%d")
    url = f"https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/{year}/him_sst_pac_D{date_str}.txt"
    
    print(f"Fetching HIMAWARI SST data from: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading HIMAWARI data: {e}")
        return None

def parse_himawari_sst(text):
    """
    Parses the 601-record HIMAWARI SST text file.
    Output: (metadata, grid_data)
    Grid: 600 rows (60N-0N) x 800 cols (100E-180E)
    """
    if not text:
        return None, None
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 601:
        print(f"Warning: Unexpected file length ({len(lines)} lines)")
        return None, None
    
    # Record 1: Header (Year Month Day)
    header = lines[0].split()
    metadata = {
        "date": " ".join(header),
        "lat_range": (0.0, 60.0),
        "lon_range": (100.0, 180.0),
        "resolution": 0.1
    }
    
    # Records 2-601: Data
    grid = []
    for line in lines[1:601]:  # type: ignore
        # The data consists of 800 values, each 3 digits wide.
        # Spacing may be inconsistent. Remove all spaces.
        clean_line = line.replace(" ", "").replace("\t", "").replace("\r", "")
        
        row_values = []
        # Support up to 800 values (2400 digits)
        data_len = len(clean_line)
        for i in range(0, min(data_len, 2400), 3):
            val_str = clean_line[i:i+3]
            if len(val_str) == 3:
                try:
                    row_values.append(int(val_str))
                except ValueError:
                    row_values.append(999)
        
        # Pad with 999 if row is shorter than 800 values
        if len(row_values) < 800:
            row_values.extend([999] * (800 - len(row_values)))
        elif len(row_values) > 800:
            row_values = row_values[:800]  # type: ignore
            
        grid.append(row_values)

    if len(grid) != 600:
        print(f"Warning: Grid has {len(grid)} rows instead of 600")
        return metadata, None

    # Convert to numpy array
    grid_np = np.array(grid, dtype=float)
    
    # Apply Masking: Values 800 and above are flags (888: Ice, 999: Land/Unknown)
    # Also mask values outside reasonable sea temperature range (-5 to 45)
    mask = (grid_np >= 800)
    grid_np[mask] = np.nan
    grid_np = grid_np * 0.1
    
    # Secondary mask for physical plausibility
    grid_np[(grid_np < -5) | (grid_np > 45)] = np.nan
    
    return metadata, grid_np

if __name__ == "__main__":
    # Test with a recent date (e.g., yesterday or 2026-01-01)
    test_date = datetime(2026, 1, 1)
    raw = fetch_himawari_sst_data(test_date)
    meta, grid = parse_himawari_sst(raw)
    if grid is not None:
        print(f"Successfully parsed grid: {grid.shape}")
        # Use nanmean, nanmin, nanmax safely
        print(f"Stats - Min: {np.nanmin(grid):.1f}°C, Max: {np.nanmax(grid):.1f}°C, Mean: {np.nanmean(grid):.1f}°C")
