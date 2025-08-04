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

## 🗂️ Project Structure

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

### 📊 Exploratory Data Analysis

🔍 **Observation:**

## 🏆 Final Submission Result

---

## 🧾 Conclusion

This project successfully developed an air quality prediction framework for **Ho Chi Minh City**, integrating data visualization as a core component of the machine learning pipeline. The study combined robust data preprocessing, feature engineering, and XGBoost-based modeling to address environmental data analysis challenges.

### 🔑 Key Achievements:

- **Data Integration**: Successfully merged PM10 data from 6 monitoring stations with meteorological data (2019–2023), implementing IQR-based outlier detection and time-based feature engineering.
- **Visualization-Driven Workflow**: Applied comprehensive visualization throughout the pipeline, ensuring data quality and providing interpretable insights for stakeholders.
- **Predictive Modeling**: Deployed optimized XGBoost regression, achieving reliable PM10 forecasting through early stopping and regularization techniques.

This framework bridges complex machine learning analyses with actionable environmental health decisions, supporting public health officials and urban planners in **Vietnam's dynamic urban environment**.
