import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

def plot_pm10_distribution(df, output_dir):
    """Plot PM10 value distribution"""
    print("📊 Creating PM10 distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PM10 Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Overall distribution
    axes[0, 0].hist(df['pm10'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Overall PM10 Distribution')
    axes[0, 0].set_xlabel('PM10 (µg/m³)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['pm10'].mean(), color='red', linestyle='--', label=f'Mean: {df["pm10"].mean():.1f}')
    axes[0, 0].axvline(df['pm10'].median(), color='green', linestyle='--', label=f'Median: {df["pm10"].median():.1f}')
    axes[0, 0].legend()
    
    # Box plot by season
    if 'season' in df.columns:
        sns.boxplot(data=df, x='season', y='pm10', ax=axes[0, 1])
        axes[0, 1].set_title('PM10 Distribution by Season')
        axes[0, 1].set_ylabel('PM10 (µg/m³)')
    
    # Log-scale distribution
    pm10_positive = df['pm10'][df['pm10'] > 0].dropna()
    axes[1, 0].hist(np.log10(pm10_positive), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('PM10 Distribution (Log Scale)')
    axes[1, 0].set_xlabel('Log10(PM10)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Top stations by average PM10
    station_means = df.groupby('station_code')['pm10'].mean().sort_values(ascending=False).head(10)
    axes[1, 1].bar(range(len(station_means)), station_means.values, color='orange', alpha=0.7)
    axes[1, 1].set_title('Top 10 Stations by Average PM10')
    axes[1, 1].set_xlabel('Station Rank')
    axes[1, 1].set_ylabel('Average PM10 (µg/m³)')
    axes[1, 1].set_xticks(range(len(station_means)))
    axes[1, 1].set_xticklabels([f'{i+1}' for i in range(len(station_means))])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pm10_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_missing_data_patterns(df, output_dir):
    """Visualize missing data patterns"""
    print("📊 Creating missing data pattern plots...")
    
    # Calculate missing data percentage by column
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Missing Data Patterns Analysis', fontsize=16, fontweight='bold')
    
    # Missing data by feature
    features_with_missing = missing_percent[missing_percent > 0].sort_values(ascending=True)
    if len(features_with_missing) > 0:
        axes[0].barh(range(len(features_with_missing)), features_with_missing.values, color='red', alpha=0.7)
        axes[0].set_yticks(range(len(features_with_missing)))
        axes[0].set_yticklabels(features_with_missing.index)
        axes[0].set_xlabel('Missing Percentage (%)')
        axes[0].set_title('Missing Data by Feature')
        axes[0].grid(axis='x', alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No Missing Data Found!', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[0].transAxes, fontsize=14, color='green')
        axes[0].set_title('Missing Data by Feature')
    
    # Missing data by station
    station_missing = df.groupby('station_code')['pm10'].apply(lambda x: x.isnull().sum() / len(x) * 100)
    station_missing = station_missing[station_missing > 0].sort_values(ascending=False)
    
    if len(station_missing) > 0:
        top_missing_stations = station_missing.head(20)
        axes[1].bar(range(len(top_missing_stations)), top_missing_stations.values, color='orange', alpha=0.7)
        axes[1].set_xticks(range(len(top_missing_stations)))
        axes[1].set_xticklabels(top_missing_stations.index, rotation=45, ha='right')
        axes[1].set_ylabel('Missing Percentage (%)')
        axes[1].set_title('Top 20 Stations with Missing PM10 Data')
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Missing PM10 Data by Station!', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=axes[1].transAxes, fontsize=14, color='green')
        axes[1].set_title('Missing PM10 Data by Station')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_data_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_trends(df, output_dir):
    """Show temporal trends"""
    print("📊 Creating temporal trend plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('PM10 Temporal Trends Analysis', fontsize=16, fontweight='bold')
    
    # Monthly trend
    monthly_avg = df.groupby('month')['pm10'].mean()
    axes[0, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('Average PM10 by Month')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Average PM10 (µg/m³)')
    axes[0, 0].set_xticks(range(1, 13))
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hourly trend
    hourly_avg = df.groupby('hour')['pm10'].mean()
    axes[0, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6, color='green')
    axes[0, 1].set_title('Average PM10 by Hour of Day')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Average PM10 (µg/m³)')
    axes[0, 1].set_xticks(range(0, 24, 3))
    axes[0, 1].grid(True, alpha=0.3)
    
    # Daily trend (day of week)
    daily_avg = df.groupby('dayofweek')['pm10'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), daily_avg.values, color='orange', alpha=0.7)
    axes[1, 0].set_title('Average PM10 by Day of Week')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Average PM10 (µg/m³)')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(day_names)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Yearly trend
    yearly_avg = df.groupby('year')['pm10'].mean()
    axes[1, 1].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=3, markersize=10, color='red')
    axes[1, 1].set_title('Average PM10 by Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Average PM10 (µg/m³)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_station_statistics(df, output_dir):
    """Display station-wise statistics"""
    print("📊 Creating station statistics plots...")
    
    # Calculate station statistics
    station_stats = df.groupby('station_code')['pm10'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    station_stats = station_stats.sort_values('mean', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Station-wise PM10 Statistics', fontsize=16, fontweight='bold')
    
    # Top 15 stations by average PM10
    top_stations = station_stats.head(15)
    axes[0, 0].barh(range(len(top_stations)), top_stations['mean'], color='red', alpha=0.7)
    axes[0, 0].set_yticks(range(len(top_stations)))
    axes[0, 0].set_yticklabels(top_stations['station_code'])
    axes[0, 0].set_xlabel('Average PM10 (µg/m³)')
    axes[0, 0].set_title('Top 15 Stations by Average PM10')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Station data completeness
    station_stats['completeness'] = (station_stats['count'] / station_stats['count'].max()) * 100
    complete_stations = station_stats.sort_values('completeness', ascending=False).head(15)
    axes[0, 1].barh(range(len(complete_stations)), complete_stations['completeness'], color='green', alpha=0.7)
    axes[0, 1].set_yticks(range(len(complete_stations)))
    axes[0, 1].set_yticklabels(complete_stations['station_code'])
    axes[0, 1].set_xlabel('Data Completeness (%)')
    axes[0, 1].set_title('Top 15 Stations by Data Completeness')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # PM10 variability (coefficient of variation)
    station_stats['cv'] = station_stats['std'] / station_stats['mean']
    variable_stations = station_stats.sort_values('cv', ascending=False).head(15)
    axes[1, 0].barh(range(len(variable_stations)), variable_stations['cv'], color='orange', alpha=0.7)
    axes[1, 0].set_yticks(range(len(variable_stations)))
    axes[1, 0].set_yticklabels(variable_stations['station_code'])
    axes[1, 0].set_xlabel('Coefficient of Variation')
    axes[1, 0].set_title('Top 15 Most Variable Stations')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Scatter plot: Mean vs Standard Deviation
    axes[1, 1].scatter(station_stats['mean'], station_stats['std'], alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Mean PM10 (µg/m³)')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('PM10 Mean vs Standard Deviation by Station')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'station_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed station statistics
    station_stats.to_csv(output_dir / 'station_detailed_stats.csv', index=False)
    print(f"💾 Detailed station statistics saved to: {output_dir / 'station_detailed_stats.csv'}")

def plot_correlation_heatmap(df, output_dir):
    """Create correlation heatmap of features"""
    print("📊 Creating correlation heatmap...")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove datetime-related object columns and keep main features
    feature_cols = ['pm10', 'year', 'month', 'day', 'hour', 'dayofweek', 'dayofyear',
                   'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                   'is_weekend', 'is_rush_hour']
    
    # Add lag and rolling features if they exist
    lag_cols = [col for col in numeric_cols if 'lag' in col or 'rolling' in col]
    feature_cols.extend(lag_cols)
    
    # Keep only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(feature_cols) < 2:
        print("⚠️ Not enough numeric features for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation matrix
    corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')
    print(f"💾 Correlation matrix saved to: {output_dir / 'correlation_matrix.csv'}")
    
    # Print top correlations with PM10
    pm10_corr = corr_matrix['pm10'].drop('pm10').abs().sort_values(ascending=False)
    print("\n🔍 Top 10 features correlated with PM10:")
    for i, (feature, corr_val) in enumerate(pm10_corr.head(10).items(), 1):
        print(f"  {i:2d}. {feature:<20}: {corr_val:.3f}")

def create_comprehensive_report(df, output_dir, processing_stats):
    """Create a comprehensive data analysis report"""
    print("📊 Creating comprehensive analysis report...")
    
    # Basic statistics
    total_records = len(df)
    unique_stations = df['station_code'].nunique()
    date_range = f"{df['datetime'].min()} to {df['datetime'].max()}"
    
    # PM10 statistics
    pm10_stats = df['pm10'].describe()
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    # Station statistics
    station_stats = df.groupby('station_code')['pm10'].agg(['count', 'mean', 'std']).reset_index()
    station_stats.columns = ['station_code', 'record_count', 'avg_pm10', 'std_pm10']
    
    # Time coverage analysis
    time_coverage = df.groupby('station_code')['datetime'].agg(['min', 'max', 'count']).reset_index()
    time_coverage['days_covered'] = (time_coverage['max'] - time_coverage['min']).dt.days
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PM10 Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .stat-box {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; min-width: 200px; text-align: center; }}
            .stat-number {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 PM10 Air Quality Data Analysis Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Dataset Overview</h2>
            <div class="stat-box">
                <div class="stat-number">{total_records:,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{unique_stations}</div>
                <div class="stat-label">Monitoring Stations</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(df.columns)}</div>
                <div class="stat-label">Features</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{pm10_stats['mean']:.1f}</div>
                <div class="stat-label">Average PM10 (µg/m³)</div>
            </div>
            
            <div class="highlight">
                <strong>📅 Time Period:</strong> {date_range}
            </div>
            
            <h2>📈 PM10 Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value (µg/m³)</th></tr>
                <tr><td>Mean</td><td>{pm10_stats['mean']:.2f}</td></tr>
                <tr><td>Median</td><td>{pm10_stats['50%']:.2f}</td></tr>
                <tr><td>Standard Deviation</td><td>{pm10_stats['std']:.2f}</td></tr>
                <tr><td>Minimum</td><td>{pm10_stats['min']:.2f}</td></tr>
                <tr><td>Maximum</td><td>{pm10_stats['max']:.2f}</td></tr>
                <tr><td>25th Percentile</td><td>{pm10_stats['25%']:.2f}</td></tr>
                <tr><td>75th Percentile</td><td>{pm10_stats['75%']:.2f}</td></tr>
            </table>
            
            <h2>📋 Data Processing Summary</h2>
            <table>
                <tr><th>File</th><th>Original Rows</th><th>Final Rows</th><th>Stations</th><th>Processing Rate</th></tr>
    """
    
    for fname, stats in processing_stats.items():
        processing_rate = (stats['final_rows'] / stats['original_rows']) * 100 if stats['original_rows'] > 0 else 0
        html_content += f"""
                <tr>
                    <td>{fname}</td>
                    <td>{stats['original_rows']:,}</td>
                    <td>{stats['final_rows']:,}</td>
                    <td>{stats['stations']}</td>
                    <td>{processing_rate:.1f}%</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <h2>🏭 Top 10 Stations by Average PM10</h2>
            <table>
                <tr><th>Rank</th><th>Station Code</th><th>Average PM10 (µg/m³)</th><th>Records</th></tr>
    """
    
    top_stations = station_stats.nlargest(10, 'avg_pm10')
    for i, (_, row) in enumerate(top_stations.iterrows(), 1):
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['station_code']}</td>
                    <td>{row['avg_pm10']:.2f}</td>
                    <td>{row['record_count']:,}</td>
                </tr>
        """
    
    # Data quality section
    features_with_missing = missing_percent[missing_percent > 0]
    html_content += f"""
            </table>
            
            <h2>🔍 Data Quality Assessment</h2>
            <div class="stat-box">
                <div class="stat-number">{(missing_percent == 0).sum()}</div>
                <div class="stat-label">Complete Features</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(features_with_missing)}</div>
                <div class="stat-label">Features with Missing Data</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{missing_percent.mean():.2f}%</div>
                <div class="stat-label">Average Missing Rate</div>
            </div>
    """
    
    if len(features_with_missing) > 0:
        html_content += """
            <h3>Missing Data by Feature</h3>
            <table>
                <tr><th>Feature</th><th>Missing Count</th><th>Missing Percentage</th></tr>
        """
        for feature, pct in features_with_missing.items():
            count = missing_data[feature]
            html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{count:,}</td>
                    <td>{pct:.2f}%</td>
                </tr>
            """
        html_content += "</table>"
    
    html_content += """
            <h2>📊 Generated Visualizations</h2>
            <ul>
                <li><strong>pm10_distribution.png</strong> - PM10 value distribution analysis</li>
                <li><strong>missing_data_patterns.png</strong> - Missing data visualization</li>
                <li><strong>temporal_trends.png</strong> - Time-based trend analysis</li>
                <li><strong>station_statistics.png</strong> - Station-wise performance metrics</li>
                <li><strong>correlation_heatmap.png</strong> - Feature correlation analysis</li>
            </ul>
            
            <div class="highlight">
                <strong>💡 Recommendations:</strong>
                <ul>
                    <li>Monitor stations with high PM10 variability for data quality issues</li>
                    <li>Consider seasonal adjustments in forecasting models</li>
                    <li>Investigate stations with incomplete data coverage</li>
                    <li>Use highly correlated features for predictive modeling</li>
                </ul>
            </div>
            
            <hr>
            <p style="text-align: center; color: #7f8c8d; font-size: 12px;">
                Report generated by PM10 Data Preprocessing Pipeline
            </p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = output_dir / 'analysis_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Comprehensive analysis report saved to: {report_path}")

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
    plots_dir = output_dir / "visualizations"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Generate all visualizations
        print("\n🎨 GENERATING VISUALIZATIONS")
        print("=" * 50)
        
        plot_pm10_distribution(final_df, plots_dir)
        plot_missing_data_patterns(final_df, plots_dir)
        plot_temporal_trends(final_df, plots_dir)
        plot_station_statistics(final_df, plots_dir)
        plot_correlation_heatmap(final_df, plots_dir)
        
        # Create comprehensive report
        create_comprehensive_report(final_df, output_dir, processing_stats)
        
        # Save feature information
        feature_info = pd.DataFrame({
            'feature': final_df.columns,
            'dtype': [str(dtype) for dtype in final_df.dtypes],
            'non_null_count': [final_df[col].count() for col in final_df.columns],
            'null_percentage': [f"{(final_df[col].isnull().sum() / len(final_df) * 100):.2f}%" 
                               for col in final_df.columns]
        })
        
        feature_info.to_csv(output_dir / "feature_info.csv", index=False)
        print(f"\n💾 Feature information saved to: {output_dir / 'feature_info.csv'}")
        
        print(f"\n✅ All data successfully processed and visualized!")
        print(f"📁 Check the '{plots_dir}' folder for all visualization files")
        print(f"📄 Open '{output_dir / 'analysis_report.html'}' for the comprehensive report")
        
    else:
        print("⚠️ No data was successfully processed. Please check input files.")

if __name__ == "__main__":
    main()