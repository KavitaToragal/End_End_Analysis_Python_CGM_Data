# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

st.set_page_config(page_title="HUPA-UC Diabetes Dashboard", layout="wide")

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Team7_DataDynamos_cleaned_data.xlsx")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    return df

df = load_data().copy()

# -------------------- Sidebar --------------------
st.sidebar.header("🔍 Filters")
patients = df["Patient ID"].unique()
selected_patient = st.sidebar.selectbox("Select Patient", ["All"] + list(patients))
year_filter = st.sidebar.multiselect(
    "Select Year(s)", df['time'].dt.year.unique(),
    default=df['time'].dt.year.unique()
)

df_filtered = df.copy()
if selected_patient != "All":
    df_filtered = df_filtered[df_filtered["Patient ID"] == selected_patient]
df_filtered = df_filtered[df_filtered['time'].dt.year.isin(year_filter)]

if df_filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# -------------------- Title & KPIs --------------------
st.title("📊 HUPA-UC Diabetes Dashboard")
st.markdown(
    "A unified view of continuous glucose, lifestyle, and intervention data for Type 1 Diabetes management."
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df_filtered):,}")
col2.metric("Avg Glucose", f"{df_filtered['glucose'].mean():.1f} mg/dL")
col3.metric("Unique Patients", df_filtered["Patient ID"].nunique())
st.markdown("---")

# -------------------- Main Tabs --------------------
tab_desc, tab_presc, tab_pred = st.tabs(["📈 Descriptive", "💊 Prescriptive", "🤖 Predictive"])

# -------------------- Descriptive Tab --------------------
with tab_desc:
    st.header("📈 Descriptive Analysis")

    # 1. Glucose distribution
    st.markdown("**1. Glucose Distribution** – Overall glucose levels compared to hypo/hyper thresholds.")
    fig, ax = plt.subplots(figsize=(3, 1.5))
    sns.histplot(df_filtered['glucose'], bins=40, kde=True, color="skyblue", ax=ax)
    ax.axvline(70, color='red', linestyle='--', label='Hypo <70')
    ax.axvline(180, color='green', linestyle='--', label='Hyper >180')
    ax.set_xlabel("Glucose (mg/dL)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Most glucose readings cluster in normal range, with clear hypo and hyper outliers.")

    # 2. Time in Range
    st.markdown("**2. Time in Range (TIR)** – Proportion of glucose readings within, above, or below range.")
    tir_counts = df_filtered['glucose'].apply(
        lambda x: 'Hypo' if x < 70 else ('Normal' if x <= 180 else 'Hyper')
    ).value_counts()
    st.bar_chart(tir_counts, height=150)
    st.markdown("**Insight:** Time in Range metric gives a quick overview of glucose control quality.")

    # 3. Hourly glucose trends
    st.markdown("**3. Hourly Glucose Trends** – Average glucose variation across hours of the day.")
    df_filtered['hour'] = df_filtered['time'].dt.hour
    hourly_glucose = df_filtered.groupby('hour')['glucose'].mean()
    fig, ax = plt.subplots(figsize=(3, 1.5))
    hourly_glucose.plot(marker='o', color="orange", ax=ax)
    ax.set_xlabel("Hour of Day", fontsize=8)
    ax.set_ylabel("Avg Glucose", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Shows circadian glucose rhythms, useful for detecting dawn effect or post-meal spikes.")

    # 4. Carb intake by meal period
    st.markdown("**4. Carb Intake by Meal Period** – Average carb intake at breakfast, lunch, dinner, and snacks.")
    def meal_period(hour):
        if 6 <= hour < 11: return "Breakfast"
        elif 11 <= hour < 15: return "Lunch"
        elif 17 <= hour < 21: return "Dinner"
        else: return "Other"
    df_filtered["meal_period"] = df_filtered["hour"].apply(meal_period)
    carb_by_meal = df_filtered.groupby("meal_period")["carb_input"].mean()
    fig, ax = plt.subplots(figsize=(3, 1.5))
    carb_by_meal.plot(kind="bar", color="teal", ax=ax)
    ax.set_ylabel("Avg Carb (g)", fontsize=8)
    ax.set_xlabel("Meal Period", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Higher carb intake is typically observed during main meals.")

    # 5. Steps per patient
    st.markdown("**5. Steps per Patient** – Average steps per patient across the cohort.")
    steps_avg = df_filtered.groupby("Patient ID")["steps"].mean()
    fig, ax = plt.subplots(figsize=(3, 1.5))
    steps_avg.plot(kind="bar", color="lightgreen", ax=ax)
    ax.set_ylabel("Avg Steps", fontsize=8)
    ax.set_xlabel("Patient ID", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Physical activity varies widely across patients, impacting glucose control.")

    # 6. Bolus insulin per patient
    st.markdown("**6. Bolus Insulin per Patient** – Average bolus insulin delivered.")
    bolus_avg = df_filtered.groupby("Patient ID")["bolus_volume_delivered"].mean()
    fig, ax = plt.subplots(figsize=(3, 1.5))
    bolus_avg.plot(kind="bar", color="purple", ax=ax)
    ax.set_ylabel("Avg Bolus (units)", fontsize=8)
    ax.set_xlabel("Patient ID", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Bolus insulin needs differ per patient, reflecting eating patterns and insulin sensitivity.")

    # 7. Basal insulin per patient
    st.markdown("**7. Basal Insulin per Patient** – Average basal insulin delivery.")
    basal_avg = df_filtered.groupby("Patient ID")["basal_rate"].mean()
    fig, ax = plt.subplots(figsize=(3, 1.5))
    basal_avg.plot(kind="bar", color="navy", ax=ax)
    ax.set_ylabel("Avg Basal (units/hr)", fontsize=8)
    ax.set_xlabel("Patient ID", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Basal insulin is relatively stable across patients compared to bolus doses.")

    st.markdown("---")
    st.success("Descriptive insights highlight glucose patterns, lifestyle behaviors, and insulin use across patients.")

# -------------------- Prescriptive Tab --------------------
with tab_presc:
    st.header("💊 Prescriptive Analysis")

    df_filtered['time'] = pd.to_datetime(df_filtered['time'])
    df_filtered['hour'] = df_filtered['time'].dt.hour
    df_filtered['date'] = df_filtered['time'].dt.date
    df_filtered['year'] = df_filtered['time'].dt.year

    patient_list = df_filtered['Patient ID'].unique()
    selected_patient = st.selectbox("Select Patient", patient_list)
    patient_data = df_filtered[df_filtered['Patient ID'] == selected_patient]

    # --- Q9 ---
    st.markdown("### Q9: Quarterly Trends")
    patient_data['quarter'] = patient_data['time'].dt.to_period('Q')
    factors = ['steps', 'calories', 'heart_rate', 'basal_rate']
    quarterly_avg = patient_data.groupby('quarter').agg({
        'glucose':'mean','steps':'mean','calories':'mean',
        'heart_rate':'mean','basal_rate':'mean'
    }).reset_index()
    quarters = quarterly_avg['quarter'].astype(str)

    fig, axes = plt.subplots(len(factors),1, figsize=(6,2*len(factors)), sharex=True)
    for i,factor in enumerate(factors):
        ax1 = axes[i]
        ax1.plot(quarters, quarterly_avg['glucose'], marker='o', color='tab:red', label='Glucose')
        ax2 = ax1.twinx()
        ax2.plot(quarters, quarterly_avg[factor], marker='o', color='tab:blue', label=factor)
        ax1.set_title(f'Glucose vs {factor}')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Quarterly patterns highlight factor-glucose relationships over time.")

    # --- Q10 ---
    st.markdown("### Q10: Glucose Distribution per Patient")
    plt.figure(figsize=(6,2))
    sns.boxplot(x='Patient ID', y='glucose', data=df_filtered, palette='viridis')
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())
    st.markdown("**Insight:** Some patients have higher glucose variability than others.")

    # --- Q11 ---
    st.markdown("### Q11: Daily Average Glucose (Selected Year)")
    selected_year = st.selectbox("Select Year", sorted(patient_data['year'].unique()))
    df_year = patient_data[patient_data['year']==selected_year].groupby('date')['glucose'].mean()
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(df_year.index, df_year.values, marker='o', markersize=3)
    st.pyplot(fig)
    st.markdown("**Insight:** Daily average glucose trends highlight fluctuations over the year.")

    # --- Q12 ---
    st.markdown("### Q12: TIR/TAR/TBR Metrics")
    low, high = 70, 180
    total = len(patient_data)
    tir = (patient_data['glucose'].between(low, high).sum()/total)*100
    tar = (patient_data['glucose']>high).sum()/total*100
    tbr = (patient_data['glucose']<low).sum()/total*100
    fig, ax = plt.subplots(figsize=(4,2))
    ax.bar(['TIR','TAR','TBR'], [tir,tar,tbr], color=['green','red','blue'])
    st.pyplot(fig)
    st.markdown("**Insight:** Most patients spend majority of time in range, but TAR remains a challenge.")

    # --- Q13 ---
    st.markdown("### Q13: Daily TIR/TAR/TBR with Factors")
    daily_metrics = (
        patient_data.groupby('date').agg(
            TIR=('glucose', lambda x: x.between(low, high).mean() * 100),
            TAR=('glucose', lambda x: (x > high).mean() * 100),
            TBR=('glucose', lambda x: (x < low).mean() * 100),
        ).reset_index()
    )
    factors_to_plot = ['steps','calories','heart_rate','basal_rate','carb_input','bolus_volume_delivered']
    fig, axes = plt.subplots(len(factors_to_plot),1, figsize=(6,2*len(factors_to_plot)), sharex=True)
    for i,factor in enumerate(factors_to_plot):
        ax=axes[i]
        ax.stackplot(daily_metrics['date'], daily_metrics['TIR'], daily_metrics['TAR'], daily_metrics['TBR'],
                     labels=['TIR','TAR','TBR'], colors=['#A8E6CF','#FFD3B6','#FFAAA5'])
        if factor in patient_data.columns:
            daily_factor = patient_data.groupby('date')[factor].mean()
            ax2=ax.twinx()
            ax2.plot(daily_factor.index, daily_factor.values, color='black', marker='o', markersize=3)
        ax.set_title(f"TIR/TAR/TBR vs {factor}")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Insight:** Daily lifestyle/insulin factors directly impact glucose control metrics.")

    # --- Q14 ---
    st.markdown("### Q14: Glucose Correlation with Factors")
    corr_factors = ['steps','calories','heart_rate','basal_rate','carb_input','bolus_volume_delivered']
    corr_matrix = patient_data[['glucose']+corr_factors].corr().loc['glucose', corr_factors]
    st.bar_chart(corr_matrix)
    st.markdown("**Insight:** Correlation identifies key drivers of glucose variation per patient.")

    # --- Q15 ---
    st.markdown("### Q15: Dawn Effect (3AM–8AM)")
    dawn_df = patient_data[(patient_data['hour']>=3)&(patient_data['hour']<=8)]
    dawn_curve = dawn_df.groupby('hour')['glucose'].mean()
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(dawn_curve.index, dawn_curve.values, marker='o', color='black')
    ax.axhline(100, color='blue', linestyle='--')
    ax.axhspan(80,120, color='green', alpha=0.2)
    st.pyplot(fig)
    st.markdown("**Insight:** Dawn effect highlights early morning rises in glucose.")

    # --- Q16 ---
    st.markdown("### Q16: Meal Glucose AUC")
    meal_windows = {"Breakfast":(6,10),"Lunch":(11,15),"Dinner":(18,22)}
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=list(meal_windows.keys()))
    for i,(meal,(start,end)) in enumerate(meal_windows.items(),1):
        df_meal = patient_data[(patient_data['hour']>=start)&(patient_data['hour']<end)]
        if len(df_meal)>0:
            x=(df_meal['time']-df_meal['time'].dt.normalize()).dt.total_seconds()/60
            y=df_meal['glucose']
            auc = np.trapz(y,x)
            fig.add_trace(go.Scatter(x=x,y=y,mode='lines',fill='tozeroy', name=f"{meal} AUC={auc:.1f}"), row=1,col=i)
    fig.update_layout(height=300, width=900)
    st.plotly_chart(fig)
    st.markdown("**Insight:** AUC quantifies post-meal glucose excursions.")

# -------------------- Predictive Tab --------------------
with tab_pred:
    st.header("🤖 Predictive Analysis: High Glucose Risk")

    diabts_df = df_filtered.copy()
    diabts_df['hour_of_day'] = diabts_df['time'].dt.hour
    diabts_df['day_of_week'] = diabts_df['time'].dt.dayofweek
    diabts_df['high_glucose'] = (diabts_df['glucose']>180).astype(int)

    features = ['Age','Gender','Race','Average Sleep Duration (hrs)','Sleep Quality (1-10)',
                '% with Sleep Disturbances','steps','calories','heart_rate','basal_rate',
                'bolus_volume_delivered','carb_input','hour_of_day','day_of_week']

    X = diabts_df[features]
    y = diabts_df['high_glucose']

    numeric_features = ['Age','Average Sleep Duration (hrs)','Sleep Quality (1-10)','% with Sleep Disturbances',
                        'steps','calories','heart_rate','basal_rate','bolus_volume_delivered','carb_input',
                        'hour_of_day','day_of_week']
    categorical_features = ['Gender','Race']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False,
                                  eval_metric='logloss', random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=3)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Neural Network", MLPClassifier(max_iter=200, random_state=42))
    ]

    results_list = []
    plt.figure(figsize=(7,5))
    for name, model in models:
        pipe = Pipeline([('preprocess', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe,'predict_proba') else pipe.decision_function(X_test)
        fpr,tpr,_=roc_curve(y_test,y_proba)
        auc=roc_auc_score(y_test,y_proba)
        plt.plot(fpr,tpr,label=f"{name} (AUC={auc:.2f})")
        results_list.append({
            "Model":name,
            "Accuracy":accuracy_score(y_test,y_pred),
            "Precision":precision_score(y_test,y_pred),
            "Recall":recall_score(y_test,y_pred),
            "F1":f1_score(y_test,y_pred),
            "ROC AUC":auc
        })
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: High Glucose Prediction")
    plt.legend()
    st.pyplot(plt.gcf())
    st.markdown("**Insight:** ROC curves highlight trade-offs between precision and recall across models.")

    df_results=pd.DataFrame(results_list)
    numeric_cols=['Accuracy','Precision','Recall','F1','ROC AUC']
    df_results['Overall']=df_results[numeric_cols].mean(axis=1)
    st.markdown("### 🔹 Model Comparison")
    st.dataframe(df_results.style.highlight_max(subset=['Overall'], color='lightgreen'))
    st.markdown("**Insight:** The table summarizes all models, with the best overall highlighted.")
