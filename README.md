# 🔋 Model for Air Quality Forecasting– AIR PPM Hackathon 2025

## 👨‍💻 Team SigmaSquad (International University )

| Name                  | Description             |
| --------------------- | ----------------------- |
| Nguyen Thanh Nam      | Team Leader, IU student |
| Hoang Ngoc Quynh Anh | IU student              |
| Nguyen Thanh Hung     | IU student              |
| Tran Le Bao Ngoc      | IU student              |

---

## 🧠 Introduction

![AIR PPM Hackathon]()

This project was developed as part of the [AIR PPM Hackathon](https://www.airppm.org/home), hosted in Kitakyushu, Japan.
Our team, **SigmaSquad**, aimed to tackle the critical issue of develop innovative algorithms for predicting air quality using time series analysis and machine learning techniques.

The core of our solution is a machine learning pipeline utilizing \*\*\*\*, to predict air quality using historical hourly usage data.

---
### 📁 Data Preprocessing

- **`preprocess.ipynb`**  
  Loads and cleans the main energy consumption dataset.  
  It performs:
  - Data loading from raw PM10 and weather files
  - Outlier detection using the IQR method
  - Missing value handling
  - Time-based feature extraction (e.g., hour, weekday, month)
  - Output saved as a structured training dataset (`train_data.csv`)

### 🗂️ Project Setup

```bash
pip install -r requirements.txt
```
## Required Data Files for Docker file (Competition_Requirement)

# landuse.pbf
This file is required but too large for GitHub (102.87 MB).

**Download Instructions:**
- File: landuse.pbf  
- Size: ~103 MB
- Location: Place in /test/ directory
- Source: [https://download.geofabrik.de/europe/poland.html]

## 🗂️ Project Structure


```
.
├── main.py                         # Main entry point for processing data
├── models/
│   ├── scaler_xgb_X.joblib         # Scaler for input features
│   └── xgboost_pm10_model.joblib   # Trained XGBoost regression model
├── Dockerfile                      # Docker build script
├── requirements.txt                # Python dependencies
├── data/
└── test/
    ├── data.json                   # Input file (mounted during evaluation)
    └── landuse.pbf                 # Optional landuse data (mounted during evaluation)
```
## Paths Used Inside Docker

The code uses fixed absolute paths inside the container to comply with the competition requirements:

```python
scaler_X = joblib.load("/app/models/scaler_xgb_X.joblib")
MODEL = joblib.load("/app/models/xgboost_pm10_model.joblib")

with open("/app/data/test/data.json", "r") as f:
    data = json.load(f)

landuse_path = "/app/data/test/landuse.pbf"
```

⚠️ **Do not modify these paths** — they are configured to match the evaluation system setup.

## How to Build and Run

### 🐳 Build the Docker Image

To build the Docker image, run the following command in the terminal:

```bash
docker build -t airppm_submission .
```

### ▶️ Run the Docker Image (Local Test)

# 1.Correct data
To run the Docker container locally, ensure `data.json` and `landuse.pbf` are placed in the `data/test/` directory. Then execute:

```bash
docker run --rm -v "$(Get-Location):/app" -w /app my-pm10-image `
>>   --data-file data/test/data.json `
>>   --output-file output.json `
>>   --landuse-pbf data/test/landuse.pbf
```
The output will be written to `/app/output/output.json` inside the container and mirrored locally if the output directory is mounted.

### Output Format

The final output will be written to:

```
/app/output/output.json
```

Each forecast must contain 24 hourly PM₁₀ predictions per case, adhering to the exact format specified in the hackathon documentation.

# 2.Missing data
To run the Docker container locally, ensure `missing_data.json` and `landuse.pbf` are placed in the `data/test/` directory. Then execute:

```bash
docker run --rm -v "$(Get-Location):/app" -w /app my-pm10-image `
>>   --data-file data/test/missing_data.json `
>>   --output-file output.json `
>>   --landuse-pbf data/test/landuse.pbf
```

### Error Handling

If any case cannot be processed or predicted (e.g., missing `prediction_start_time`), the program will raise an error and exit with a non-zero code, as required by the rules. Example:

```python
ValueError: Invalid prediction_start_time for case 'case_0001': Invalid isoformat string: ''
```


### 📊 Exploratory Data Analysis

🔍 **Observation:**

## 🏆 Final Submission Result

---

## 🧾 Conclusion

This project successfully developed an air quality prediction framework for **Poland**, integrating data visualization as a core component of the machine learning pipeline. The study combined robust data preprocessing, feature engineering, and XGBoost-based modeling to address environmental data analysis challenges.

### 🔑 Key Achievements:

- **Data Integration**: Successfully merged PM10 data from 6 monitoring stations with meteorological data (2019–2023), implementing IQR-based outlier detection and time-based feature engineering.
- **Visualization-Driven Workflow**: Applied comprehensive visualization throughout the pipeline, ensuring data quality and providing interpretable insights for stakeholders.
- **Predictive Modeling**: Deployed optimized XGBoost regression, achieving reliable PM10 forecasting through early stopping and regularization techniques.

This framework bridges complex machine learning analyses with actionable environmental health decisions, supporting public health officials and urban planners in **Poland's dynamic urban environment**.
