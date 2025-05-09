import streamlit as st
import xgboost as xgb
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64

# Getting the directory of the script
script_directory = os.path.dirname(__file__)

# Loading the trained XGB model
model = xgb.XGBRegressor()
model.load_model(os.path.join(script_directory, 'GUI_Model.json'))

# Define the function for prediction
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return prediction[0]

# --- Custom Styling ---
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #1E90FF;
    }
    .stButton > button {
        background-color: #FF4C4C;
        color: white;
        font-weight: bold;
    }
    .stSuccess { color: #1E90FF; }
    .stAlert, .stError { color: #FF4C4C; }
    </style>
""", unsafe_allow_html=True)

# --- Load and Encode Logo ---
script_directory = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_directory, "NJIT_logo.png")

with open(logo_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode()

# --- Display Centered Logo ---
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{base64_image}' width='600'/>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown(
    "<h1 style='text-align: center; margin-top: 10px;'>Surface Chloride Concentration Prediction</h1>",
    unsafe_allow_html=True
)
# --- Instruction Note ---
st.markdown("### ℹ️ About This Tool")
st.write("""
This tool is developed to predict **surface chloride concentration [% mass]** in concrete exposed to marine environments. Predictions are made across a 30-year exposure period. 
Please ensure the mix design inputs are within realistic and acceptable ranges. This tool validates:
- **Water-to-Binder Ratio (0.3–0.7)**
- **Batch Volume (0.95–1.05 m³)**   
""")

# --- Step 1: Mix Proportions and Specific Gravities ---
st.header("Step 1: Mix Proportions and Specific Gravities")

components = [
    ("Cement", 406, 110, 519),
    ("Fine aggregate", 639, 552, 1232),
    ("Coarse aggregate", 1024, 410, 1305),
    ("Water", 215, 38.50, 311),
    ("Superplasticizer", 0, 0, 10),
    ("Fly ash", 72, 0, 239),
    ("Silica fume", 0, 0, 50),
    ("Blast furnace slag", 0, 0, 292)
]

st.markdown("**Enter Mix Proportions (kg/m³)** and their **Specific Gravities**:")

input_data = {}
sg_data = {}
col1, col2 = st.columns(2)

with col1:
    for name, default, min_val, max_val in components:
        input_data[name] = st.number_input(f"{name} [{min_val}-{max_val} kg/m³]", min_value=0.0, value=float(default))

with col2:
    for name, *_ in components:
        sg_data[name] = st.number_input(f"Specific Gravity of {name}", min_value=0.01, value=2.65 if name != "Water" else 1.0)

# --- Step 2: Validation ---
st.header("Step 2: Validation of mix proportion")
binder_mass = input_data["Cement"] + input_data["Fly ash"] + input_data["Silica fume"] + input_data["Blast furnace slag"]
water_binder_ratio = input_data["Water"] / binder_mass if binder_mass else 0

st.subheader("Water-to-Binder Ratio Check")
st.write(f"Calculated Water-to-binder Ratio: **{water_binder_ratio:.2f}**")
if not (0.3 <= water_binder_ratio <= 0.7):
    st.error("❌ Water-to-binder ratio is out of range (0.3–0.7). Please adjust mix proportions.")
    st.stop()
else:
    st.success("✅ Water-to-binder ratio is within range.")

volume = sum(input_data[k] / (sg_data[k] * 1000) for k in input_data)
st.subheader("Batch Volume Check")
st.write(f"Calculated Batch Volume: **{volume:.3f} m³**")
if not (0.95 <= volume <= 1.05):
    st.error("❌ Batch volume is out of acceptable range (0.95–1.05 m³). Please adjust proportions.")
    st.stop()
else:
    st.success("✅ Batch volume is within acceptable range.")

# --- Step 3: Environmental Parameters ---
st.header("Step 3: Environmental Parameters")

Cl_content = st.number_input("Chloride content in seawater (g/L) [Range: 13 - 27]", min_value=13.0, max_value=27.0, value=27.0)
temperature = st.number_input("Mean annual temperature (°C) [Range: 7 - 35]", min_value=7.0, max_value=35.0, value=35.0)

zone = st.selectbox("Exposure Zone", ["Tidal zone", "Splash zone", "Submerged zone"])
zone_ohe = {"Tidal zone": [1, 0, 0], "Splash zone": [0, 1, 0], "Submerged zone": [0, 0, 1]}
zone_tidal, zone_splash, zone_submerged = zone_ohe[zone]


# --- Custom Styling ---
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #1E90FF;
    }
    .stButton > button {
        color: white;
        font-weight: bold;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
    }
    div.stButton:nth-child(1) > button {
        background-color: #1E90FF !important;
    }
    .stSuccess { color: #1E90FF; }
    .stAlert, .stError { color: #FF4C4C; }
    </style>
""", unsafe_allow_html=True)



import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Step 4: Prediction ---
st.header("Step 4: Prediction")

# Predict button
if st.button("🔍 Predict Surface Chloride Concentration", key="predict_button"):

    # Exposure times
    table_exposure_times = [0.5, 1.0, 3.0, 5.0]
    graph_exposure_times = [0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    predicted_scs_for_table = []
    predicted_scs_for_graph = []

    expected_feature_order = [
        'Cement', 'Fine aggregate', 'Coarse aggregate', 'Water', 'Water-binder ratio',
        'Superplasticizer', 'Fly ash', 'Silica fume', 'Blast furnace slag',
        'Cl content in seawater', 'Annual temperature', 'Exposure time',
        'Tidal zone', 'Splash zone', 'Submerged zone'
    ]

    data_records = []

    for t in graph_exposure_times:
        row = {
            "Cement": input_data["Cement"],
            "Fine aggregate": input_data["Fine aggregate"],
            "Coarse aggregate": input_data["Coarse aggregate"],
            "Water": input_data["Water"],
            "Water-binder ratio": water_binder_ratio,
            "Superplasticizer": input_data["Superplasticizer"],
            "Fly ash": input_data["Fly ash"],
            "Silica fume": input_data["Silica fume"],
            "Blast furnace slag": input_data["Blast furnace slag"],
            "Cl content in seawater": Cl_content,
            "Annual temperature": temperature,
            "Exposure time": t,
            "Tidal zone": zone_tidal,
            "Splash zone": zone_splash,
            "Submerged zone": zone_submerged
        }
        row = {k: row[k] for k in expected_feature_order}
        df = pd.DataFrame([row])
        try:
            sc = model.predict(df)[0]
            sc_rounded = round(sc, 3)
            if t in table_exposure_times:
                predicted_scs_for_table.append(sc_rounded)
            predicted_scs_for_graph.append(sc_rounded)
            data_records.append({"Exposure Time (years)": t, "Predicted SCs (% mass)": sc_rounded})
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # --- Display Table ---
    st.subheader("Predicted Surface Chloride Concentration")

    table_html = """
    <style>
        .fancy-table {
            width: 60%;
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 15px;
        }
        .fancy-table th {
            background-color: #0073e6;
            color: white;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .fancy-table td {
            padding: 8px;
            border: 1px solid #ccc;
        }
        .fancy-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
    <table class="fancy-table">
        <thead>
            <tr>
                <th>Exposure Time [years]</th>
                <th>Predicted SCs [% mass]</th>
            </tr>
        </thead>
        <tbody>
    """
    for t, sc in zip(table_exposure_times, predicted_scs_for_table):
        table_html += f"<tr><td>{t}</td><td>{sc:.3f}</td></tr>"
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # --- Graph Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=graph_exposure_times,
        y=predicted_scs_for_graph,
        mode='lines+markers',
        line=dict(color='dodgerblue'),
        marker=dict(color='red'),
        name='Predicted SCs'
    ))
    fig.update_layout(
        title='Predicted Surface Chloride Concentration over Time',
        xaxis_title='Exposure Time [years]',
        yaxis_title='SCs [% mass]',
        xaxis=dict(
            title_font=dict(size=14, color='black', family='Arial'),
            tickfont=dict(size=12, color='black')
        ),
        yaxis=dict(
            title_font=dict(size=14, color='black', family='Arial'),
            tickfont=dict(size=12, color='black')
        ),
        template='plotly_white'
    )
    st.plotly_chart(fig)

# --- Footer ---
st.markdown("""
    <hr>
    <div style='display: flex; justify-content: space-between; color: #1E90FF; font-size: 14px;'>
        <div>
            <strong>Developed by:</strong><br>
            H. Sami Ullah<br>
            Graduate Research Assistant<br>
            MatSlab, NJIT
        </div>
        <div>
            <strong>Supervised by:</strong><br>
            <div style='display: flex; gap: 40px;'>
                <div>
                    Mathew J. Bandelt<br>
                    Associate Professor,<br>
                    Co-Director MatSlab,<br>
                    NJIT
                </div>
                <div>
                    Matthew P. Adams<br>
                    Associate Professor,<br>
                    Co-Director MatSlab,<br>
                    NJIT
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


