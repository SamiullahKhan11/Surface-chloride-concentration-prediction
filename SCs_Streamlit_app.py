import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# Load Trained Gaussian Process Models
# ==========================================================
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model: {path}\n\n{e}")
        return None

model_concrete = load_model("concrete_gpr.joblib")
model_uhpc = load_model("uhpc_gpr.joblib")

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Chloride Concentration Prediction",
    layout="wide"
)

# ==========================================================
# Custom Styling
# ==========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-size:18px !important;
}

.title-box{
    background-color:DodgerBlue;
    color:white;
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:34px !important;
    font-weight:bold;
}

.step-title{
    font-size:24px !important;
    font-weight:bold;
    color:DodgerBlue;
    margin-top:20px;
    margin-bottom:10px;
}

div[data-testid="stNumberInput"] label{
    font-size:18px !important;
    font-weight:bold;
}

.stNumberInput input{
    font-size:20px !important;
    height:55px !important;
}

/* ---------- RED BUTTON ---------- */

div.stButton > button:first-child {
    background-color: red;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 8px;
    width: 100%;
    height: 55px;
    border: none;
}

div.stButton > button:first-child:hover {
    background-color: darkred;
    color: white;
}

div.stButton > button:first-child:focus {
    background-color: darkred;
    color: white;
}

/* ------------------------------- */

.footer{
    margin-top:40px;
    text-align:center;
    font-size:16px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Title
# ==========================================================
st.markdown(
    """
    <div class="title-box">
    Machine Learning-Based Prediction of Chloride Concentration Profiles
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# Description
# ==========================================================
st.write(
"""
This application predicts **chloride concentration profiles** in both a conventional
**Concrete Deck** and a **UHPC Overlay Deck** using two independently developed
**Gaussian Process Regression (GPR)** machine learning models.

The application allows different exposure periods for the two systems, enabling
comparison of chloride ingress at different service ages.

The predictions are generated using previously trained Gaussian Process Regression
models and therefore no retraining is required during execution.

### Input Requirements

- Enter the desired **exposure period** for the **Concrete Deck**.
- Enter the desired **exposure period** for the **UHPC Overlay Deck**.
- Exposure time must be between **1 and 50 years**.
"""
)

# ==========================================================
# Step 1
# ==========================================================
st.markdown(
    '<div class="step-title">Step 1: Enter Exposure Time</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    concrete_year = st.number_input(
        label="Concrete Deck Exposure Time (Years) [1–50]",
        min_value=1,
        max_value=50,
        value=35,
        step=1,
        help="Allowed range: 1–50 years"
    )

with col2:

    uhpc_year = st.number_input(
        label="UHPC Overlay Deck Exposure Time (Years) [1–50]",
        min_value=1,
        max_value=50,
        value=35,
        step=1,
        help="Allowed range: 1–50 years"
    )

# ==========================================================
# Display User Inputs
# ==========================================================
st.markdown(
    '<div class="step-title">Selected Prediction Periods</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.success(f"Concrete Deck = **{concrete_year} years**")

with col2:
    st.success(f"UHPC Overlay Deck = **{uhpc_year} years**")

# ==========================================================
# Check Models
# ==========================================================
if model_concrete is not None and model_uhpc is not None:
    st.success("✅ Both Gaussian Process Regression models loaded successfully.")
else:
    st.error("One or more trained models could not be loaded.")


# ==========================================================
# Predict Chloride Profiles
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button(
        "Predict Chloride Profiles",
        key="predict_btn",
        use_container_width=True
    )

if predict:

    # --------------------------------------------
    # Define prediction depth
    # --------------------------------------------
    max_depth = 250          # Change if your specimens are deeper
    n_points = 150

    depth_prediction = np.linspace(0, max_depth, n_points)

    # --------------------------------------------
    # Create prediction data
    # --------------------------------------------
    prediction_data_con = pd.DataFrame({
        "Depth": depth_prediction,
        "Time": concrete_year
    })

    prediction_data_uhpc = pd.DataFrame({
        "Depth": depth_prediction,
        "Time": uhpc_year
    })

    # --------------------------------------------
    # Model predictions
    # --------------------------------------------
    prediction_con, std_con = model_concrete.predict(
        prediction_data_con,
        return_std=True
    )

    prediction_uhpc, std_uhpc = model_uhpc.predict(
        prediction_data_uhpc,
        return_std=True
    )

    # --------------------------------------------
    # Plot
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(12,8))

    con_line, = ax.plot(
        depth_prediction,
        prediction_con,
        color="red",
        linewidth=3,
        label=f"Concrete ({concrete_year} Years)"
    )

    ax.fill_between(
        depth_prediction,
        prediction_con-1.96*std_con,
        prediction_con+1.96*std_con,
        color="red",
        alpha=0.18
    )

    uhpc_line, = ax.plot(
        depth_prediction,
        prediction_uhpc,
        color="dodgerblue",
        linewidth=3,
        label=f"UHPC Overlay ({uhpc_year} Years)"
    )

    ax.fill_between(
        depth_prediction,
        prediction_uhpc-1.96*std_uhpc,
        prediction_uhpc+1.96*std_uhpc,
        color="dodgerblue",
        alpha=0.18
    )

    # Critical chloride
    critical_cl = 0.0006

    crit_line = ax.axhline(
        critical_cl,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Critical Chloride"
    )

    ax.text(
        max_depth*0.15,
        critical_cl+0.0001,
        r"$C_{critical}=0.0006\ mol/m^3$",
        fontsize=15,
        bbox=dict(facecolor="white", edgecolor="none")
    )

    ax.set_xlim(0, max_depth)

    ax.set_xlabel(
        "Depth from Exposed Surface (mm)",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Chloride Concentration (mol/m³)",
        fontsize=18,
        fontweight="bold"
    )

    ax.legend(fontsize=14)

    ax.grid(alpha=0.2)

    st.pyplot(fig)

# ==========================================================
# Footer
# ==========================================================
st.markdown(
    '<div class="footer">--- End of Step 1 ---</div>',
    unsafe_allow_html=True
)

#streamlit run "c:\Users\hu32\Desktop\Ensemble ML\Chloride Content\GUI\Rheo_Streamlit_app.py"

#cd "c:/Users/hu32/Desktop/Ensemble ML/Chloride Content/GUI"
#streamlit run Rheo_Streamlit_app.py
#streamlit run GUI_NJDoT.py
