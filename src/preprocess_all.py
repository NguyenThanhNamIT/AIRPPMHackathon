import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_stations(stations_file):
    try:
        with open(stations_file, 'rb') as f:
            header = f.read(4)
            if header == b'PK\x03\x04':  
                print("  🔍 Stations file is actually Excel format")
                try:
                    stations_df = pd.read_excel(stations_file)
                    print(f"✅ Loaded stations data: {len(stations_df)} stations (Excel format)")
                    return stations_df
                except Exception as e:
                    print(f"Failed to read as Excel: {e}")
                    return None
        
        for sep in [',', ';', '\t']:
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    stations_df = pd.read_csv(stations_file, encoding=encoding, sep=sep)
                    if stations_df.shape[1] > 1:  # More than 1 column means successful parsing
                        print(f"✅ Loaded stations data: {len(stations_df)} stations (encoding: {encoding}, sep: '{sep}')")
                        return stations_df
                except:
                    continue
        
        print(f"❌ Could not load stations file in any format")
        
    except Exception as e:
        print(f"❌ Failed to load stations file: {e}")
        return None

def detect_file_format(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    separators = [',', ';', '\t']
    
    if file_path.suffix.lower() == '.csv':
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=5)
                    if df.shape[1] >= 2:  
                        print(f"  📋 Detected: CSV with {encoding} encoding, '{sep}' separator")
                        full_df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        
                        print(f"  🔍 Columns: {list(full_df.columns[:5])}...")
                        print(f"  🔍 First row values: {full_df.iloc[0].head().tolist()}")
                        
                        return full_df
                except Exception as e:
                    continue
    
    # Try Excel format
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(file_path)
            print(f"  📋 Detected: Excel file")
            print(f"  🔍 Columns: {list(df.columns[:5])}...")
            print(f"  🔍 First row values: {df.iloc[0].head().tolist()}")
            return df
        except Exception as e:
            print(f"  ❌ Excel read failed: {e}")
    
    return None

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower()
    
    datetime_patterns = ['thời gian', 'time', 'datetime', 'date', 'ngày', 'thoigian']
    
    for col in df.columns:
        for pattern in datetime_patterns:
            if pattern in col:
                df.rename(columns={col: 'datetime'}, inplace=True)
                break
    
    if 'datetime' not in df.columns:
        df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
        print("  ⚠️ No datetime column detected, using first column")
    
    return df

def debug_datetime_column(df):
    print(f"  🔍 Datetime column info:")
    print(f"    - Column name: '{df.columns[0] if 'datetime' in df.columns else 'datetime not found'}'")
    print(f"    - Sample values: {df['datetime'].head(5).tolist()}")
    print(f"    - Data type: {df['datetime'].dtype}")
    print(f"    - Unique sample: {df['datetime'].unique()[:10]}")

def parse_datetime(df):
  
    debug_datetime_column(df)
    
    datetime_formats = [
        '%Y-%m-%dT%H:%M:%S',     
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', 
        '%d/%m/%Y %H:%M',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y',
        '%Y%m%d %H:%M:%S',
        '%Y%m%d',
        '%d.%m.%Y %H:%M:%S',
        '%d.%m.%Y %H:%M',
        '%d.%m.%Y'
    ]
    
    original_count = len(df)
    original_datetime = df['datetime'].copy()
    
    for fmt in datetime_formats:
        try:
            test_conversion = pd.to_datetime(original_datetime, format=fmt, errors='coerce')
            valid_count = test_conversion.notna().sum()
            if valid_count > original_count * 0.8:  
                df['datetime'] = test_conversion
                print(f"  📅 Parsed {valid_count}/{original_count} dates with format: {fmt}")
                break
        except Exception as e:
            continue
    
    if df['datetime'].dtype == 'object': 
        print("  🔄 Trying pandas auto-detection...")
        
        for dayfirst in [True, False]:
            try:
                test_conversion = pd.to_datetime(original_datetime, errors='coerce', dayfirst=dayfirst, infer_datetime_format=True)
                valid_count = test_conversion.notna().sum()
                if valid_count > original_count * 0.8:
                    df['datetime'] = test_conversion
                    print(f"  📅 Auto-parsed {valid_count}/{original_count} dates (dayfirst={dayfirst})")
                    break
            except:
                continue
    
    if df['datetime'].dtype == 'object':
        print("  🔄 Trying regex extraction...")
        import re
        
        def extract_datetime(text):
            try:
                text = str(text)
                
                patterns = [
                    r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', 
                    r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
                    r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})',
                    r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})',
                    r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})',
                    r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2})',
                    r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})',
                    r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2})',
                    r'(\d{2}/\d{2}/\d{4})',
                    r'(\d{2}-\d{2}-\d{4})',
                    r'(\d{4}-\d{2}-\d{2})',
                    r'(\d{4}/\d{2}/\d{2})'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
            except:
                pass
            return text  
        
        extracted = original_datetime.apply(extract_datetime)
        
        for fmt in datetime_formats:
            try:
                test_conversion = pd.to_datetime(extracted, format=fmt, errors='coerce')
                valid_count = test_conversion.notna().sum()
                if valid_count > original_count * 0.8:
                    df['datetime'] = test_conversion
                    print(f"  📅 Regex extracted and parsed {valid_count} dates with format: {fmt}")
                    break
            except:
                continue
    
    if df['datetime'].dtype == 'object':
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    valid_count = df['datetime'].notna().sum()
    print(f"  ✅ Final valid dates: {valid_count}/{original_count} ({valid_count/original_count*100:.1f}%)")
    print(f"  ✅ Datetime column type: {df['datetime'].dtype}")
    
    before_drop = len(df)
    df = df.dropna(subset=['datetime'])
    after_drop = len(df)
    
    if before_drop != after_drop:
        print(f"  🗑️ Removed {before_drop - after_drop} rows with invalid dates")
    
    return df

def process_pm10_data(df):
   
    id_vars = ['datetime']
    value_vars = [col for col in df.columns if col != 'datetime']
    
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='station_code',
        value_name='pm10'
    )
    
    df_long['station_code'] = df_long['station_code'].str.strip()
    
    df_long['pm10'] = pd.to_numeric(df_long['pm10'], errors='coerce')
    
    before_filter = len(df_long)
    df_long = df_long[(df_long['pm10'] >= 0) & (df_long['pm10'] <= 1000)]
    after_filter = len(df_long)
    
    if before_filter != after_filter:
        print(f"  🧹 Filtered out {before_filter - after_filter} invalid PM10 values")
    
    return df_long

def add_time_features(df):
    df = df.copy()
    
    if df['datetime'].dtype == 'object':
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    df = df.dropna(subset=['datetime'])
    
    if len(df) == 0:
        print("  ❌ No valid datetime values for feature extraction")
        return df
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofyear'] = df['datetime'].dt.dayofyear
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    print(f"  ✅ Added time features. Data shape: {df.shape}")
    
    return df

def add_lag_features(df, lags=[1, 2, 3, 6, 12, 24]):
    df = df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
    
    for lag in lags:
        df[f'pm10_lag_{lag}'] = df.groupby('station_code')['pm10'].shift(lag)
    
    for window in [3, 6, 12, 24]:
        df[f'pm10_rolling_mean_{window}'] = df.groupby('station_code')['pm10'].rolling(
            window=window, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        df[f'pm10_rolling_std_{window}'] = df.groupby('station_code')['pm10'].rolling(
            window=window, min_periods=1
        ).std().reset_index(0, drop=True)
    
    return df

def handle_missing_values(df, method='interpolate'):
    print(f"  🔍 Missing values before handling: {df.isnull().sum().sum()}")
    
    if method == 'interpolate':
        df = df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
        
        def interpolate_group(group):
            group_indexed = group.set_index('datetime')
            
            group_indexed['pm10'] = group_indexed['pm10'].interpolate(
                method='time', limit=3
            )
            
            return group_indexed.reset_index()
        
        df_list = []
        for station_code, group in df.groupby('station_code'):
            interpolated_group = interpolate_group(group)
            df_list.append(interpolated_group)
        
        df = pd.concat(df_list, ignore_index=True)
    
    elif method == 'forward_fill':
        df = df.sort_values(['station_code', 'datetime'])
        df['pm10'] = df.groupby('station_code')['pm10'].fillna(method='ffill', limit=2)
    
    elif method == 'mean_fill':
        hourly_means = df.groupby(['station_code', 'hour'])['pm10'].mean()
        df['pm10'] = df.apply(
            lambda row: hourly_means.get((row['station_code'], row['hour']), row['pm10']) 
            if pd.isna(row['pm10']) else row['pm10'], 
            axis=1
        )
    
    print(f"  ✅ Missing values after handling: {df.isnull().sum().sum()}")
    return df

def remove_outliers(df, method='iqr', factor=1.5):
    before_count = len(df)
    
    if method == 'iqr':
        Q1 = df.groupby('station_code')['pm10'].quantile(0.25)
        Q3 = df.groupby('station_code')['pm10'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bounds = Q1 - factor * IQR
        upper_bounds = Q3 + factor * IQR
        
        df = df[
            (df.apply(lambda row: row['pm10'] >= lower_bounds[row['station_code']], axis=1)) &
            (df.apply(lambda row: row['pm10'] <= upper_bounds[row['station_code']], axis=1))
        ]
    
    after_count = len(df)
    print(f"  🎯 Removed {before_count - after_count} outliers ({(before_count - after_count)/before_count*100:.1f}%)")
    
    return df

def main():
    pm10_files = [
        "2019.csv",
        "2020_PM10_1g.xlsx", 
        "2020.csv",
        "2021_PM10_1g.xlsx",
        "2021.csv", 
        "2022_PM10_1g.xlsx",
        "2022.csv",
        "2023_PM10_1g.xlsx", 
        "2023.csv"
    ]
    
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stations_df = load_stations(input_dir / "Stations.csv")
    
    all_data = []
    processing_stats = {}
    
    print("🚀 Starting PM10 data preprocessing...\n")
    
    for fname in pm10_files:
        file_path = input_dir / fname
        print(f"📥 Processing {file_path.name}")
        
        if not file_path.exists():
            print(f"  ❌ File not found: {file_path}")
            continue
        
        df = detect_file_format(file_path)
        if df is None:
            print(f"  ❌ Failed to load {file_path.name}")
            continue
        
        original_shape = df.shape
        print(f"  📊 Original shape: {original_shape}")
        
        df = clean_column_names(df)
        df = parse_datetime(df)
        df = process_pm10_data(df)
        
        if len(df) == 0:
            print(f"  ❌ No valid data after processing {file_path.name}")
            continue
        
        df = add_time_features(df)
        
        df = handle_missing_values(df, method='interpolate')
        df = remove_outliers(df, method='iqr')
        
        df = add_lag_features(df)
        
        df = df.dropna()
        
        before_dup = len(df)
        df = df.drop_duplicates(subset=['datetime', 'station_code'])
        after_dup = len(df)
        
        if before_dup != after_dup:
            print(f"  🗑️ Removed {before_dup - after_dup} duplicate records")
        
        final_shape = df.shape
        print(f"  ✅ Final shape: {final_shape}")
        
        processing_stats[fname] = {
            'original_rows': original_shape[0],
            'final_rows': final_shape[0],
            'stations': df['station_code'].nunique(),
            'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}"
        }
        
        all_data.append(df)
        print()
    
    if all_data:
        print("🔄 Combining all datasets...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        final_df = final_df.sort_values(['station_code', 'datetime']).reset_index(drop=True)
        
        output_file = output_dir / "train_data.csv"
        final_df.to_csv(output_file, index=False)
        
        total_records = len(final_df)
        unique_stations = final_df['station_code'].nunique()
        date_range = f"{final_df['datetime'].min()} to {final_df['datetime'].max()}"
        
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total records: {total_records:,}")
        print(f"Unique stations: {unique_stations}")
        print(f"Date range: {date_range}")
        print(f"Features: {len(final_df.columns)}")
        print(f"Output file: {output_file}")
        
        print("\n📋 Per-file statistics:")
        for fname, stats in processing_stats.items():
            print(f"  {fname}: {stats['original_rows']:,} → {stats['final_rows']:,} rows "
                  f"({stats['stations']} stations)")
        
        feature_info = pd.DataFrame({
            'feature': final_df.columns,
            'dtype': [str(dtype) for dtype in final_df.dtypes],
            'non_null_count': [final_df[col].count() for col in final_df.columns],
            'null_percentage': [f"{(final_df[col].isnull().sum() / len(final_df) * 100):.2f}%" 
                               for col in final_df.columns]
        })
        
        feature_info.to_csv(output_dir / "feature_info.csv", index=False)
        print(f"\n💾 Feature information saved to: {output_dir / 'feature_info.csv'}")
        
        print(f"\n✅ All data successfully processed and saved!")
        
    else:
        print("⚠️ No data was successfully processed. Please check input files.")

if __name__ == "__main__":
    main()

