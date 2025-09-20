HUPA-UC (Human-Centered Predictive Analytics – Unified Care) Diabetes Project
Overview

This project leverages Continuous Glucose Monitoring (CGM) and lifestyle data from individuals with Type 1 Diabetes to generate descriptive, prescriptive, and predictive insights. The dataset integrates CGM, clinical, lifestyle, sleep, dietary, and demographic information, enabling analysis of glucose dynamics, lifestyle patterns, and their interrelations.

Population: 25 individuals with Type 1 Diabetes

Duration: ≥ 14 days per individual

Devices Used:

Freestyle Libre 2 CGMs → glucose data (5-minute intervals)

Fitbit Ionic → heart rate, sleep, physical activity

Self-reporting → insulin doses, carbohydrate intake

Demographics → age, gender, race, sleep quality

Project Goal

To clean, integrate, and analyze multi-modal patient data for dual-risk management of glucose levels and cardiovascular strain, while enabling actionable interventions and predictive alerts.

Repository Structure
1. Raw Data

HUPA-UC Diabetes Dataset/

25 patient CSV files (HUPA0001P.csv … HUPA0028P.csv)

T1DM_patient_sleep_demographics_with_race.csv

Key variables:

time, glucose (mg/dL), carb_input (g), bolus_volume_delivered (insulin U),
basal_rate (U/hr), heart_rate (BPM), calories, steps

Demographics: Age, Gender, Race, Sleep Duration, Sleep Quality, % Disturbances

2. Data Processing

Data_Cleaning_Preprocessing.ipynb → column standardization, timestamp fixes, missing value imputation, outlier removal

Final Excel File Integration Script.ipynb → merges all patient files + demographics → diabetes_cleaned.xlsx

Key Cleaning Rules:

carb_input: 0 → missing

basal_rate: <0.05 or >5 removed

glucose: <40 or >500 flagged

heart_rate: <40 or >180 flagged

bolus_volume: <0.5 or >100 removed

steps/calories: negative or zero removed

time: split into date + time
