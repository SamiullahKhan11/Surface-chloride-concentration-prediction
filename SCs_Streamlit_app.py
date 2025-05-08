import streamlit as st
import xgboost as xgb
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# --- Logos and Title (Improved Layout) ---
col1, col2 = st.columns([1, 1])
with col1:
    st.image(os.path.join(script_directory, "NJIT_logo.png"), width=150)
with col2:
    st.image(os.path.join(script_directory, "MatSlab_logo.png"), width=150)

st.markdown("<h1 style='text-align: center; margin-top: 10px;'>Surface Chloride Concentration Prediction</h1>", unsafe_allow_html=True)

# --- Instruction Note ---
st.markdown("### ℹ️ About This Tool")
st.write("""
This tool is developed to predict **surface chloride concentration [% mass]** in concrete exposed to marine environments. 
Please ensure the mix design inputs are within realistic and acceptable ranges. This tool validates:
- **Water-to-Binder Ratio (0.3–0.7)**
- **Batch Volume (0.95–1.05 m³)**  
Predictions are made across a 30-year exposure period.
""")

# --- Step 1: Mix Proportions and Specific Gravities ---
st.header("Step 1: Mix Proportions and Specific Gravities")

components = [
    ("Cement", 325, 25, 45),
    ("Fine aggregate", 650, 500, 900),
    ("Coarse aggregate", 1050, 800, 1200),
    ("Water", 180, 120, 250),
    ("Superplasticizer", 8, 0, 20),
    ("Fly ash", 50, 0, 100),
    ("Silica fume", 10, 0, 50),
    ("Blast furnace slag", 30, 0, 150)
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
binder_mass = input_data["Cement"] + input_data["Fly ash"] + input_data["Silica fume"] + input_data["Blast furnace slag"]
water_binder_ratio = input_data["Water"] / binder_mass if binder_mass else 0

st.subheader("Water-to-Binder Ratio Check")
st.write(f"Calculated W/B Ratio: **{water_binder_ratio:.2f}**")
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

Cl_content = st.number_input("Chloride content in seawater (g/L)", min_value=0.0, value=19.0)
temperature = st.number_input("Mean annual temperature (°C)", min_value=-10.0, value=25.0)

zone = st.selectbox("Exposure Zone", ["Tidal zone", "Splash zone", "Submerged zone"])
zone_ohe = {"Tidal zone": [1, 0, 0], "Splash zone": [0, 1, 0], "Submerged zone": [0, 0, 1]}
zone_tidal, zone_splash, zone_submerged = zone_ohe[zone]

# --- Step 4: Prediction ---
st.header("Step 4: Prediction")

exposure_times = np.arange(0.5, 30, 2.5)
predicted_scs = []

# Feature order based on the model's training data
expected_feature_order = [
    'Cement', 'Fine aggregate', 'Coarse aggregate', 'Water', 'Water-binder ratio', 'Superplasticizer', 'Fly ash',
    'Silica fume', 'Blast furnace slag', 'Cl content in seawater', 'Annual temperature', 'Exposure time',
    'Tidal zone', 'Splash zone', 'Submerged zone'
]

# Correcting the feature order for prediction
for t in exposure_times:
    row = {
        "Cement": input_data["Cement"],
        "Fine aggregate": input_data["Fine aggregate"],
        "Coarse aggregate": input_data["Coarse aggregate"],
        "Water": input_data["Water"],
        "Water-binder ratio": water_binder_ratio,  # Add the calculated Water-binder ratio here
        "Superplasticizer": input_data["Superplasticizer"],
        "Fly ash": input_data["Fly ash"],
        "Silica fume": input_data["Silica fume"],
        "Blast furnace slag": input_data["Blast furnace slag"],
        "Cl content in seawater": Cl_content,
        "Annual temperature": temperature,  # Using "Annual temperature" as the feature name
        "Exposure time": t,
        "Tidal zone": zone_tidal,
        "Splash zone": zone_splash,
        "Submerged zone": zone_submerged
    }
    
    # Reorder the features to match the model's expected order
    row = {k: row[k] for k in expected_feature_order}

    # Create the DataFrame for prediction
    df = pd.DataFrame([row])
    
    # Prediction with the loaded model
    try:
        sc = model.predict(df)[0]
        predicted_scs.append(sc)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        break

# --- Plotting ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=exposure_times,
    y=predicted_scs,
    mode='lines+markers',
    line=dict(color='dodgerblue'),
    marker=dict(color='red'),
    name='Predicted SC'
))
fig.update_layout(
    title='Predicted Surface Chloride Concentration over Time',
    xaxis_title='Exposure Time [years]',
    yaxis_title='Surface Chloride Concentration [% mass]',
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
            Research Graduate Assistant<br>
            MatSlab, NJIT
        </div>
        <div>
            <strong>Supervised by:</strong><br>
            Mathew J. Bandelt – Associate Professor<br>
                                Co-Director MatSlab<br>
                                NJIT<br>
            Matthew P. Adams – Associate Professor<br>
                                Co-Director MatSlab<br>
                                NJIT
        </div>
    </div>
""", unsafe_allow_html=True)



#streamlit run "c:/Users/hu32/Desktop/Ensemble ML/Chloride Content/GUI/SCs_Streamlit_app.py"
