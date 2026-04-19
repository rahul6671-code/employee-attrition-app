
# ============================================================
# Employee Attrition Prediction — Streamlit App
# Based on IBM HR Analytics Dataset
# Compatible with Google Colab + Streamlit (via localtunnel/ngrok)
# ============================================================

# ── INSTALL (run this cell in Colab before launching) ───────
# !pip install streamlit shap imbalanced-learn xgboost lightgbm pyngrok --quiet

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

import shap
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .risk-high   { background:#ffe0e0; border-left:6px solid #e74c3c; padding:12px; border-radius:6px; margin:6px 0; }
    .risk-medium { background:#fff8e0; border-left:6px solid #f39c12; padding:12px; border-radius:6px; margin:6px 0; }
    .risk-low    { background:#e0f8e0; border-left:6px solid #2ecc71; padding:12px; border-radius:6px; margin:6px 0; }
    h1 { color: #2c3e50; }
    .stButton>button { background:#2c3e50; color:white; border-radius:8px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("👥 Employee Attrition Prediction System")
st.markdown("**AI/ML Project** | IBM HR Analytics | Upload your CSV and get instant attrition predictions")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    use_smote = st.checkbox("Apply SMOTE (handle class imbalance)", value=True)
    selected_models = st.multiselect(
        "Models to Train",
        ["Logistic Regression", "Decision Tree", "Random Forest",
         "XGBoost", "LightGBM", "Gradient Boosting"],
        default=["Random Forest", "XGBoost", "LightGBM"]
    )
    st.markdown("---")
    st.info("📌 Upload the IBM HR Analytics CSV to begin.\nDataset available on Kaggle or GitHub.")

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.header("📂 Step 1: Upload Dataset")

col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded_file = st.file_uploader(
        "Upload employee dataset (CSV format)",
        type=["csv"],
        help="Expected format: IBM HR Analytics dataset with an 'Attrition' column (Yes/No)"
    )

with col_up2:
    st.markdown("#### 💡 Expected Columns")
    st.code("""Age, Attrition, Department,
Gender, JobRole, MonthlyIncome,
OverTime, JobSatisfaction,
YearsAtCompany, ...""")

# ─────────────────────────────────────────────
# LOAD DATA — uploaded OR default GitHub
# ─────────────────────────────────────────────
@st.cache_data
def load_default():
    url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
    return pd.read_csv(url)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded: **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
else:
    st.info("No file uploaded yet. Using the default IBM HR dataset from GitHub for demo.")
    try:
        df = load_default()
        st.success(f"✅ Default dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Could not load default dataset: {e}")
        st.stop()

if 'Attrition' not in df.columns:
    st.error("❌ Dataset must contain an 'Attrition' column (Yes/No). Please upload the correct file.")
    st.stop()

# ─────────────────────────────────────────────
# EDA SECTION
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 Step 2: Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Overview", "Attrition Analysis", "Feature Distributions"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", df.shape[0])
    c2.metric("Total Features", df.shape[1])
    attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
    c3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    c4.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Data Types Summary")
    dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    st.dataframe(dtype_df, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Attrition'].value_counts()
        ax.bar(counts.index, counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax.set_title('Attrition Count', fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.05))
        ax.set_title('Attrition Proportion', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Categorical breakdowns
    cat_features = [c for c in ['Department', 'JobRole', 'OverTime', 'MaritalStatus', 'BusinessTravel', 'Gender'] if c in df.columns]
    if cat_features:
        st.subheader("Attrition Rate by Category")
        sel_cat = st.selectbox("Select feature:", cat_features)
        fig, ax = plt.subplots(figsize=(9, 4))
        ct = pd.crosstab(df[sel_cat], df['Attrition'], normalize='index') * 100
        ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
        ax.set_title(f'Attrition % by {sel_cat}', fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['No Attrition', 'Attrition'])
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)
        plt.close()

with tab3:
    num_cols_available = [c for c in ['Age', 'MonthlyIncome', 'DistanceFromHome',
                                       'TotalWorkingYears', 'YearsAtCompany',
                                       'YearsSinceLastPromotion', 'JobSatisfaction'] if c in df.columns]
    if num_cols_available:
        sel_num = st.selectbox("Select numerical feature:", num_cols_available)
        fig, ax = plt.subplots(figsize=(9, 4))
        yes = df[df['Attrition'] == 'Yes'][sel_num]
        no  = df[df['Attrition'] == 'No'][sel_num]
        ax.hist(no,  bins=25, alpha=0.6, color='#2ecc71', label='No Attrition')
        ax.hist(yes, bins=25, alpha=0.6, color='#e74c3c', label='Attrition')
        ax.set_title(f'Distribution of {sel_num} by Attrition', fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────
# PREPROCESSING + TRAINING
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🤖 Step 3: Train Models")

@st.cache_resource
def train_pipeline(df_raw, test_size, use_smote, selected_models):
    df_model = df_raw.copy()

    # Drop low-value columns
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], inplace=True)

    # Encode target
    df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)

    # Binary encode
    for col in ['Gender', 'OverTime']:
        if col in df_model.columns:
            df_model[col] = df_model[col].map({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0})

    # One-hot
    cat_cols = df_model.select_dtypes(include='object').columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    # Feature engineering
    if 'YearsAtCompany' in df_model.columns and 'TotalWorkingYears' in df_model.columns:
        df_model['CompanyLoyaltyRatio'] = df_model['YearsAtCompany'] / (df_model['TotalWorkingYears'] + 1)
    if 'YearsSinceLastPromotion' in df_model.columns and 'YearsAtCompany' in df_model.columns:
        df_model['PromotionGap'] = df_model['YearsSinceLastPromotion'] / (df_model['YearsAtCompany'] + 1)

    X = df_model.drop('Attrition', axis=1)
    y = df_model['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled  = scaler.transform(X_test)

    all_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
        'LightGBM':            LGBMClassifier(random_state=42, verbose=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name in selected_models:
        if name not in all_models:
            continue
        model = all_models[name]
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train_sm)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_sm, y_train_sm)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1':        f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC':   roc_auc_score(y_test, y_prob),
        }

    best_model_name = max(results, key=lambda n: results[n]['AUC-ROC'])

    return results, best_model_name, X_test, y_test, X_train_sm, scaler

if not selected_models:
    st.warning("Please select at least one model in the sidebar.")
    st.stop()

with st.spinner("🔄 Training models... this may take 30–60 seconds."):
    results, best_model_name, X_test, y_test, X_train_sm, scaler = train_pipeline(
        df, test_size, use_smote, selected_models
    )

st.success(f"✅ Training complete! Best model: **{best_model_name}** (AUC = {results[best_model_name]['AUC-ROC']:.4f})")

# ─────────────────────────────────────────────
# MODEL RESULTS
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📈 Step 4: Model Results")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
scores_df = pd.DataFrame({name: {m: results[name][m] for m in metrics} for name in results}).T.round(4)
scores_df.index.name = "Model"

# Highlight best row
def highlight_best(row):
    return ['background-color: #d4edda; font-weight: bold' if row.name == best_model_name else '' for _ in row]

st.subheader("📋 Metrics Table")
st.dataframe(scores_df.style.apply(highlight_best, axis=1), use_container_width=True)

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.subheader("📊 AUC-ROC Comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors_bar = ['#e74c3c' if n == best_model_name else '#3498db' for n in results]
    bars = ax.bar(list(results.keys()), [results[n]['AUC-ROC'] for n in results],
                  color=colors_bar, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model AUC-ROC Comparison', fontweight='bold')
    for bar, val in zip(bars, [results[n]['AUC-ROC'] for n in results]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    plt.xticks(rotation=20, ha='right')
    st.pyplot(fig)
    plt.close()

with col_r2:
    st.subheader("📉 ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
    for (name, res), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} ({res['AUC-ROC']:.3f})", color=color, linewidth=2)
    ax.plot([0,1],[0,1],'k--', linewidth=1.2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    st.pyplot(fig)
    plt.close()

# Confusion Matrix
st.subheader(f"🔢 Confusion Matrix — {best_model_name}")
best = results[best_model_name]
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stay', 'Leave'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
st.pyplot(fig)
plt.close()

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
best_clf = best['model']
if hasattr(best_clf, 'feature_importances_'):
    st.subheader("🔑 Top 15 Attrition Drivers")
    feat_imp = pd.Series(best_clf.feature_importances_, index=X_train_sm.columns).sort_values(ascending=False)
    top15 = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_feat = plt.cm.RdYlGn_r(np.linspace(0, 1, 15))
    ax.barh(range(15), top15.values[::-1], color=colors_feat)
    ax.set_yticks(range(15))
    ax.set_yticklabels(top15.index[::-1], fontsize=9)
    ax.set_title(f'Top 15 Feature Importances — {best_model_name}', fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# BULK PREDICTION ON TEST SET
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🚨 Step 5: At-Risk Employees (Test Set)")

test_results_df = X_test.copy()
test_results_df['Actual_Attrition'] = y_test.values
test_results_df['Attrition_Probability'] = best['y_prob']
test_results_df['Risk_Level'] = test_results_df['Attrition_Probability'].apply(
    lambda x: '🔴 High' if x >= 0.70 else ('🟡 Medium' if x >= 0.40 else '🟢 Low')
)

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("🔴 High Risk",   (test_results_df['Risk_Level'] == '🔴 High').sum())
col_b2.metric("🟡 Medium Risk", (test_results_df['Risk_Level'] == '🟡 Medium').sum())
col_b3.metric("🟢 Low Risk",    (test_results_df['Risk_Level'] == '🟢 Low').sum())

st.subheader("Top 10 Highest-Risk Employees")
top10 = test_results_df.sort_values('Attrition_Probability', ascending=False).head(10)
display_cols = ['Attrition_Probability', 'Risk_Level', 'Actual_Attrition']
for col in ['MonthlyIncome', 'OverTime', 'JobSatisfaction']:
    if col in top10.columns:
        display_cols.insert(-1, col)
st.dataframe(top10[display_cols].style.format({'Attrition_Probability': '{:.2%}'}), use_container_width=True)

# Download button
csv_download = test_results_df[display_cols].to_csv(index=False)
st.download_button("⬇️ Download Full Risk Report (CSV)", csv_download,
                   file_name="attrition_risk_report.csv", mime="text/csv")

# ─────────────────────────────────────────────
# SINGLE EMPLOYEE PREDICTOR
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🔍 Step 6: Predict for a Single Employee")
st.markdown("Fill in the employee details below and click **Predict**.")

with st.form("employee_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age          = st.slider("Age", 18, 60, 30)
        gender       = st.selectbox("Gender", ["Male", "Female"])
        marital      = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        department   = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_level    = st.slider("Job Level (1–5)", 1, 5, 2)

    with col2:
        monthly_inc  = st.number_input("Monthly Income ($)", 1000, 25000, 5000, 500)
        job_sat      = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
        env_sat      = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
        wlb          = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
        overtime     = st.selectbox("OverTime", ["No", "Yes"])

    with col3:
        yrs_company  = st.slider("Years at Company", 0, 40, 5)
        yrs_promo    = st.slider("Years Since Last Promotion", 0, 15, 1)
        total_yrs    = st.slider("Total Working Years", 0, 40, 8)
        distance     = st.slider("Distance From Home (km)", 1, 30, 10)
        stock        = st.slider("Stock Option Level (0–3)", 0, 3, 1)
        job_inv      = st.slider("Job Involvement (1–4)", 1, 4, 3)

    submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

if submitted:
    employee_data = {
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital,
        'Department': department,
        'JobLevel': job_level,
        'MonthlyIncome': monthly_inc,
        'JobSatisfaction': job_sat,
        'EnvironmentSatisfaction': env_sat,
        'WorkLifeBalance': wlb,
        'OverTime': overtime,
        'YearsAtCompany': yrs_company,
        'YearsSinceLastPromotion': yrs_promo,
        'TotalWorkingYears': total_yrs,
        'DistanceFromHome': distance,
        'StockOptionLevel': stock,
        'JobInvolvement': job_inv,
    }

    emp_df = pd.DataFrame([employee_data])
    if 'Gender'   in emp_df.columns: emp_df['Gender']   = emp_df['Gender'].map({'Male': 1, 'Female': 0})
    if 'OverTime' in emp_df.columns: emp_df['OverTime'] = emp_df['OverTime'].map({'Yes': 1, 'No': 0})
    emp_df = pd.get_dummies(emp_df)
    emp_df = emp_df.reindex(columns=X_train_sm.columns, fill_value=0)

    clf  = best_clf
    prob = clf.predict_proba(emp_df)[0][1]
    pred = clf.predict(emp_df)[0]

    if prob >= 0.70:
        risk = "🔴 HIGH RISK"
        css_class = "risk-high"
        action = "Immediate 1-on-1 meeting, compensation review, explore flexible work options"
    elif prob >= 0.40:
        risk = "🟡 MEDIUM RISK"
        css_class = "risk-medium"
        action = "Career development plan, training opportunities, manager check-in"
    else:
        risk = "🟢 LOW RISK"
        css_class = "risk-low"
        action = "Maintain engagement with recognition and growth opportunities"

    st.markdown(f"""
    <div class="{css_class}">
        <h3>{risk}</h3>
        <b>Attrition Probability:</b> {prob:.1%}<br>
        <b>Prediction:</b> {"⚠️ Will Leave" if pred == 1 else "✅ Will Stay"}<br>
        <b>Recommended HR Action:</b> {action}
    </div>
    """, unsafe_allow_html=True)

    # Probability gauge
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([0], [prob], color='#e74c3c' if prob >= 0.7 else '#f39c12' if prob >= 0.4 else '#2ecc71', height=0.5)
    ax.barh([0], [1 - prob], left=[prob], color='#ecf0f1', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Attrition Probability')
    ax.set_title(f'Risk Score: {prob:.1%}', fontweight='bold')
    ax.axvline(0.40, color='orange', linestyle='--', linewidth=1.2, label='Medium threshold')
    ax.axvline(0.70, color='red',    linestyle='--', linewidth=1.2, label='High threshold')
    ax.legend(fontsize=8, loc='upper right')
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# BUSINESS IMPACT
# ─────────────────────────────────────────────
st.markdown("---")
st.header("💼 Step 7: Business Impact Estimator")

total_emps   = st.number_input("Total Employees in Company", 100, 50000, 1470, 100)
avg_salary   = st.number_input("Average Annual Salary ($)", 20000, 200000, 65000, 5000)
retention_rt = st.slider("Retention Success Rate (% of flagged employees retained)", 10, 60, 30)

attr_rate     = attrition_rate / 100
model_recall  = results[best_model_name]['Recall']
repl_cost     = avg_salary * 1.2

leaving       = int(total_emps * attr_rate)
total_cost    = leaving * repl_cost
detected      = int(leaving * model_recall)
retained      = int(detected * (retention_rt / 100))
savings       = retained * repl_cost

c1, c2, c3 = st.columns(3)
c1.metric("Estimated Employees Leaving/yr", leaving)
c2.metric("Total Attrition Cost", f"${total_cost:,.0f}")
c3.metric("💰 Estimated Savings with ML", f"${savings:,.0f}")

roi = (savings / total_cost) * 100 if total_cost > 0 else 0
st.info(f"📈 **ROI of ML Model:** {roi:.1f}% cost reduction — retaining {retained} employees/year")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#7f8c8d; font-size:0.9em'>
    Built with ❤️ using Streamlit · IBM HR Analytics Dataset · 
    Models: LR, DT, RF, XGBoost, LightGBM, GBM · SMOTE for class imbalance
</div>
""", unsafe_allow_html=True)
