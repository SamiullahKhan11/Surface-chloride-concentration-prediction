import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap

# ==========================================================
# Load Trained Gaussian Process Models
# ==========================================================
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model: {path}\n\n{e}")
        return None


# -------------------------------
# Chloride Prediction Models
# -------------------------------
model_concrete = load_model("concrete_gpr.joblib")
model_uhpc = load_model("uhpc_gpr.joblib")


# -------------------------------
# Corrosion Prediction Models
# -------------------------------
active_model_con = load_model("active_concrete.joblib")
rust_model_con = load_model("rust_concrete.joblib")

active_model_uhpc = load_model("active_uhpc.joblib")
rust_model_uhpc = load_model("rust_uhpc.joblib")

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Bridge Deck Performance Assessment and Life-Cycle Decision Support Tool",
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
    Bridge Deck Performance Assessment and Life-Cycle Decision Support Tool
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# Description
# ==========================================================
st.write(
"""
This application provides an integrated framework for evaluating the long-term
performance of reinforced concrete bridge deck systems by combining:

- **Chloride diffusion and corrosion prediction**
- **Steel deterioration assessment**
- **Life-cycle cost analysis (LCCA)**
- **Structural reliability evaluation**
- **Bridge Deck Value Index (BDVI) for decision making**

The tool compares two bridge deck alternatives:

- **Conventional Concrete Deck**
- **UHPC Overlay Deck System**

The objective is to evaluate how each system performs over its service life by
considering both **structural performance** and **economic impact**.

### Analysis Workflow

The application follows a sequential analysis process:

**Step 1: Chloride Profile Prediction**

The chloride concentration distribution through the deck depth is predicted for
both deck systems. This provides the chloride exposure information required for
subsequent corrosion calculations.

**Step 2: Corrosion and Steel Deterioration Prediction**

The chloride profiles are used to estimate corrosion initiation and steel
section loss over time. The resulting reinforcement deterioration is used as
input for structural performance evaluation.

**Step 3: Life-Cycle Cost Analysis (LCCA)**

The economic performance of each deck system is evaluated by considering:

- Initial construction cost
- Inspection costs
- Routine maintenance costs
- Rehabilitation costs
- Discounted future expenditures

Users can define different service lives, maintenance schedules, and unit costs
for each deck alternative.

**Step 4: Structural Reliability Analysis**

The effect of reinforcement deterioration on structural performance is assessed
using probabilistic analysis. The reliability index and probability of failure
are calculated considering uncertainties in:

- Material properties
- Reinforcement deterioration
- Applied loading

**Step 5: Bridge Deck Value Index (BDVI)**

The final decision-support stage combines reliability performance and economic
performance into a single index. Users can assign different importance weights
to reliability and cost depending on their design priorities.

### How to Use the Tool

Users provide project-specific information such as:

- Bridge deck geometry
- Material properties
- Environmental exposure conditions
- Service life assumptions
- Maintenance strategies
- Cost parameters
- Reliability weighting factors

The application does not store user inputs. All calculations are performed
during the current session, allowing users to explore different design
scenarios and compare alternative bridge deck systems.

### Recommended Usage

For meaningful comparisons, users should maintain consistent assumptions between
the different analysis steps. The service life, exposure conditions, and
material properties selected in earlier steps directly influence the later
corrosion, reliability, and decision-support results.

"""
)

# ====================================================================================================================
# Step 1
# ====================================================================================================================


st.markdown(
    """
    <div class="title-box">Step 1: Chloride Profile Prediction</div>
    """,
    unsafe_allow_html=True
)


st.write(
"""
This section predicts the chloride concentration profile through the bridge
deck depth for both the **Concrete Deck** and the **UHPC Overlay Deck** systems.

The chloride diffusion model is calibrated using exposure periods that represent
the expected service conditions of each deck system.

### Model Application Guidance

- The **Concrete Deck chloride profile** is developed based on a maximum exposure
  period of **50 years**.

- The **UHPC Overlay Deck chloride profile** is developed based on a maximum
  exposure period of **100 years**.

For the most reliable prediction, it is recommended that the selected exposure
time remains within these calibrated limits:

- Concrete Deck: **1–50 years**
- UHPC Overlay Deck: **1–100 years**

Predictions beyond these ranges represent extrapolation and may have increased
uncertainty.

### Chloride Profile Depth

The chloride concentration profile is evaluated from the deck surface down to:

**250 mm depth from the exposed surface**

This depth represents the region where chloride ingress and reinforcement
corrosion initiation are most critical for reinforced concrete bridge decks.

### Input Guidance

Please provide input parameters using the specified units:

- **Exposure Time:** Enter in **years**
  - Example: `50` means 50 years of chloride exposure.

- **Concrete Properties:** Enter values in the requested engineering units
  shown beside each input box.

- **Environmental parameters:** Use representative values for the actual bridge
  exposure condition.

### Important Note

The chloride concentration profile is used as the foundation for the subsequent
corrosion prediction and reliability analysis steps. Therefore, maintaining
consistent exposure assumptions between Step 1, Step 2, Step 4, and Step 5 will
provide more meaningful performance predictions.
"""
)


st.markdown(
    '<div class="step-title">Enter Exposure Time</div>',
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
        label="UHPC Overlay Deck Exposure Time (Years) [1–100]",
        min_value=1,
        max_value=100,
        value=75,
        step=1,
        help="Allowed range: 1–100 years"
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



# ====================================================================================================================
# STEP 2 - CORROSION PROFILE PREDICTION
# ====================================================================================================================

st.markdown(
    """
    <div class="title-box">Step 2: Corrosion Prediction and Steel Loss Assessment</div>
    """,
    unsafe_allow_html=True
)


st.write(
"""
This section predicts the reinforcement corrosion progression for both the
**Concrete Deck** and the **UHPC Overlay Deck** systems.

The corrosion model uses the chloride concentration profiles generated in
**Step 1** to estimate chloride exposure at the reinforcement level. The
calculated chloride concentration is then used to predict corrosion initiation,
rust expansion, and reinforcement degradation over time.

### Model Application Guidance

The corrosion prediction period should be consistent with the chloride profile
prediction range used in Step 1:

- **Concrete Deck:** Recommended prediction range = **1–50 years**
- **UHPC Overlay Deck:** Recommended prediction range = **1–100 years**

Predictions beyond these ranges may involve extrapolation of the chloride
exposure data and can introduce additional uncertainty.

### Reinforcement Corrosion Calculation

The model estimates:

- Chloride concentration at reinforcement depth
- Corrosion initiation time
- Rust expansion around reinforcing steel
- Reduction in steel cross-sectional area

The steel loss calculation is based on the reduction in reinforcement diameter
caused by corrosion products.

### Input Guidance

Please provide input values using the units shown in the input boxes:

- **Exposure Time:** Enter in **years**
  - Example: `50` represents 50 years of corrosion exposure.

- **Reinforcement Diameter:** Enter in **millimeters (mm)**
  - Example: `16` represents a 16 mm diameter reinforcing bar.

- **Reinforcement Depth/Cover:** Enter in **millimeters (mm)**
  - Example: `50` represents 50 mm concrete cover.

### Interpretation of Results

The corrosion prediction results are used in later analysis steps:

- The **Life Cycle Cost Analysis (Step 3)** evaluates economic performance.
- The **Reliability Analysis (Step 4)** uses the predicted steel degradation to
  evaluate structural performance.
- The **Bridge Deck Value Index (Step 5)** combines reliability and cost
  performance into a single decision-making indicator.

### Important Note

For meaningful comparison between the Concrete Deck and UHPC Overlay Deck,
maintain consistent reinforcement properties and geometric assumptions unless
different designs are intentionally being evaluated.
"""
)


# ==========================================================
# Steel Diameter Input
# ==========================================================

col1, col2, col3 = st.columns([1,1,1])

with col1:

    original_steel_diameter = st.number_input(
        "Original Steel Diameter (mm)",
        min_value=1.0,
        max_value=50.0,
        value=19.05,
        step=0.01
    )

    st.session_state["original_steel_diameter"] = original_steel_diameter

# ==========================================================
# Prediction Button
# ==========================================================

col1, col2, col3 = st.columns([1,2,1])

with col2:

    predict = st.button(
        "Predict Steel Corrosion Profile",
        key="rust_prediction",
        use_container_width=True
    )


# ==========================================================
# Prediction
# ==========================================================

if predict:


    # ------------------------------------------------------
    # Generate time vectors
    # ------------------------------------------------------

    years_con = np.arange(
        1,
        concrete_year + 1
    )

    years_uhpc = np.arange(
        1,
        uhpc_year + 1
    )


    input_con = pd.DataFrame({
        "Time": years_con
    })


    input_uhpc = pd.DataFrame({
        "Time": years_uhpc
    })


    # ------------------------------------------------------
    # Predict active length and rust expansion
    # ------------------------------------------------------

    active_con_all, active_std_con = active_model_con.predict(
        input_con,
        return_std=True
    )

    rust_con_all, rust_std_con = rust_model_con.predict(
        input_con,
        return_std=True
    )
    # Enforce physically realistic cumulative corrosion
    rust_con_all = np.maximum.accumulate(rust_con_all)


    active_uhpc_all, active_std_uhpc = active_model_uhpc.predict(
        input_uhpc,
        return_std=True
    )

    rust_uhpc_all, rust_std_uhpc = rust_model_uhpc.predict(
        input_uhpc,
        return_std=True
    )
    # Enforce physically realistic cumulative corrosion
    rust_uhpc_all = np.maximum.accumulate(rust_uhpc_all)


    # ======================================================
    # Create Plot Dataset
    # ======================================================

    plot_con = pd.DataFrame({

        "Time (Years)": years_con,

        "Active Bar Length (mm)": active_con_all,

        "Rust Expansion (mm)": rust_con_all,

        "Material": "Concrete Deck"

    })


    plot_uhpc = pd.DataFrame({

        "Time (Years)": years_uhpc,

        "Active Bar Length (mm)": active_uhpc_all,

        "Rust Expansion (mm)": rust_uhpc_all,

        "Material": "UHPC Overlay Deck"

    })


    plot_df = pd.concat(
        [plot_con, plot_uhpc],
        ignore_index=True
    )

    st.session_state["prediction_results_concrete"] = plot_con
    st.session_state["prediction_results_uhpc"] = plot_uhpc


    # ======================================================
    # Plot
    # ======================================================

    st.subheader(
        "Active Bar Length Reduction Due to Rust Expansion"
    )


    fig, ax = plt.subplots(
        figsize=(10,7)
    )


    custom_cmap = LinearSegmentedColormap.from_list(
        "rust_map",
        [
            "dodgerblue",
            "royalblue",
            "firebrick",
            "darkred"
        ]
    )


    sns.scatterplot(

        data=plot_df,

        x="Time (Years)",

        y="Active Bar Length (mm)",

        hue="Rust Expansion (mm)",

        style="Material",

        size="Rust Expansion (mm)",

        sizes=(70,240),

        palette=custom_cmap,

        edgecolor="black",

        linewidth=0.8,

        ax=ax

    )


    ax.set_xlabel(
        "Time [Years]",
        fontsize=20,
        fontweight="bold"
    )


    ax.set_ylabel(
        "Active Bar Length [mm]",
        fontsize=20,
        fontweight="bold"
    )


    ax.tick_params(
        labelsize=16
    )


    ax.grid(False)


    for spine in ax.spines.values():

        spine.set_linewidth(1.5)


    ax.legend(
        fontsize=14,
        frameon=True
    )


    st.pyplot(fig)


    # ======================================================
    # Steel Area Loss Calculation at Final Exposure Time
    # ======================================================

    def steel_results(rust_expansion):

        remaining_diameter = max(
            original_steel_diameter - rust_expansion,
            0
        )

        original_area = (
            np.pi * original_steel_diameter**2 / 4
        )

        remaining_area = (
            np.pi * remaining_diameter**2 / 4
        )

        area_loss = (
            original_area - remaining_area
        )

        percent_loss = (
            area_loss / original_area * 100
        )

        return (
            remaining_diameter,
            original_area,
            remaining_area,
            area_loss,
            percent_loss
        )


    # ------------------------------------------------------
    # Calculate final results
    # ------------------------------------------------------

    final_results_con = steel_results(
        rust_con_all[-1]
    )

    final_results_uhpc = steel_results(
        rust_uhpc_all[-1]
    )


    # ======================================================
    # Results Table
    # ======================================================

    st.subheader("Steel Corrosion Summary")


    table_html = f"""
    <style>

    .corrosion-table {{
        width: 85%;
        margin: 20px auto;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }}

    .corrosion-table th {{
        background-color: DodgerBlue;
        color: white;
        padding: 14px;
        text-align: center;
        border: 1px solid #D9D9D9;
        font-size: 19px;
    }}

    .corrosion-table td {{
        padding: 12px;
        text-align: center;
        border: 1px solid #D9D9D9;
    }}

    .corrosion-table tr:nth-child(even) {{
        background-color: #F7F9FC;
    }}

    .corrosion-table tr:hover {{
        background-color: #EAF3FF;
    }}

    .parameter {{
        font-weight: bold;
        text-align: left !important;
        background-color: #F2F2F2;
    }}

    </style>

    <table class="corrosion-table">

    <thead>

    <tr>
        <th>Parameter</th>
        <th>Concrete Deck</th>
        <th>UHPC Overlay Deck</th>
    </tr>

    </thead>

    <tbody>

    <tr>
        <td class="parameter">Prediction Time (Years)</td>
        <td>{concrete_year}</td>
        <td>{uhpc_year}</td>
    </tr>

    <tr>
        <td class="parameter">Original Steel Area (mm²)</td>
        <td>{final_results_con[1]:.2f}</td>
        <td>{final_results_uhpc[1]:.2f}</td>
    </tr>

    <tr>
        <td class="parameter">Remaining Steel Area (mm²)</td>
        <td>{final_results_con[2]:.2f}</td>
        <td>{final_results_uhpc[2]:.2f}</td>
    </tr>

    <tr>
        <td class="parameter">Steel Area Loss (mm²)</td>
        <td>{final_results_con[3]:.2f}</td>
        <td>{final_results_uhpc[3]:.2f}</td>
    </tr>

    <tr>
        <td class="parameter">Steel Area Loss (%)</td>
        <td><b>{final_results_con[4]:.2f}%</b></td>
        <td><b>{final_results_uhpc[4]:.2f}%</b></td>
    </tr>

    </tbody>

    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)





# ====================================================================================================================
# STEP 3 - LIFE CYCLE COST ANALYSIS (LCCA)
# ====================================================================================================================

import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D

st.markdown(
    """
    <div class="title-box">Step 3: Life Cycle Cost Analysis (LCCA)</div>
    """,
    unsafe_allow_html=True
)

st.write(
"""
This section performs a **Life Cycle Cost Analysis (LCCA)** to compare the
long-term economic performance of a conventional **Concrete Deck** and a
**UHPC Overlay Deck** system.

The LCCA estimates the total cost of each alternative throughout its selected
service life by considering both the initial construction investment and future
maintenance activities.

### Life Cycle Cost Components

The analysis includes:

- **Initial Construction Cost**
  - Calculated based on the deck geometry and material volumes.
  - For the UHPC overlay system, the total cost includes both the remaining
    concrete volume and the UHPC overlay volume.

- **Inspection Cost**
  - Represents routine bridge inspections performed at the user-defined
    inspection interval.

- **Minor Rehabilitation Cost**
  - Represents periodic maintenance activities required to maintain deck
    performance.

- **Major Rehabilitation Cost**
  - Applied only to the conventional concrete deck system.
  - UHPC overlay systems are assumed to require no major rehabilitation within
    the selected analysis period.

### Service Life Selection

The service life entered in this section is independent from the exposure periods
used in chloride diffusion and corrosion prediction.

Users can define different service lives for each alternative:

- Concrete Deck: **1–100 years**
- UHPC Overlay Deck: **1–100 years**

This allows comparison of different design scenarios, even when the two deck
systems have different expected service lives.

### Discount Rate

Future maintenance and rehabilitation costs are converted into present-day
economic value using the discount rate.

Please enter the discount rate as a decimal value:

- `0.025` means **2.5%**
- `0.05` means **5%**

Do not enter the percentage value directly.

### Input Units

Please provide all inputs using the specified engineering units:

- Deck dimensions: **meters (m)**
- Material costs: **$/m³**
- Maintenance and inspection costs: **$/m²**
- Discount rate: **decimal value**

Example:

- Concrete cost: `4000` represents **$4000/m³**
- Inspection cost: `2` represents **$2/m²**
- Discount rate: `0.025` represents **2.5%**

### Interpretation of Results

The LCCA provides:

- Cumulative life-cycle cost curves showing how costs increase over time.
- Cost breakdown showing the contribution of construction, inspection,
  maintenance, and rehabilitation activities.
- A final cost comparison between the two deck alternatives.

The LCCA results are later combined with the reliability analysis results in
**Step 5: Bridge Deck Value Index (BDVI)** to evaluate the overall performance
and economic value of each bridge deck system.

"""
)


# ==========================================================
# SERVICE LIFE INPUT
# ==========================================================

st.subheader("Service Life Input")


col1, col2 = st.columns(2)


with col1:

    lcca_concrete_year = st.number_input(
        "Concrete Deck Service Life (Years)",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Enter service life between 1 and 100 years."
    )


with col2:

    lcca_uhpc_year = st.number_input(
        "UHPC Overlay Deck Service Life (Years)",
        min_value=1,
        max_value=100,
        value=100,
        step=1,
        help="Enter service life between 1 and 100 years."
    )


# ==========================================================
# GEOMETRY INPUT
# ==========================================================

st.subheader("Bridge Deck Geometry")


col1, col2, col3 = st.columns(3)


with col1:

    deck_length = st.number_input(
        "Deck Length (m)",
        value=10.0,
        min_value=0.1,
        step=0.1,
        help="Example: 10 m"
    )


with col2:

    deck_width = st.number_input(
        "Deck Width (m)",
        value=3.3,
        min_value=0.1,
        step=0.1,
        help="Example: 3.3 m"
    )


with col3:

    total_thickness = st.number_input(
        "Total Deck Thickness (m)",
        value=0.25,
        min_value=0.01,
        step=0.01,
        help="Example: 0.25 m"
    )


st.markdown("### UHPC Overlay Thickness")


col1, col2 = st.columns(2)


with col1:

    concrete_thickness_lcca = st.number_input(
        "Concrete Thickness in UHPC System (m)",
        value=0.22,
        min_value=0.0,
        step=0.01,
        help="Example: Total thickness 0.25 m, concrete thickness 0.22 m"
    )


with col2:

    uhpc_overlay_thickness = st.number_input(
        "UHPC Overlay Thickness (m)",
        value=0.03,
        min_value=0.0,
        step=0.01,
        help="Example: 0.03 m"
    )


    # Save for later steps
    st.session_state["deck_width"] = deck_width
    st.session_state["deck_length"] = deck_length
    st.session_state["concrete_thickness_full"] = total_thickness
    st.session_state["concrete_thickness"] = concrete_thickness_lcca
    st.session_state["uhpc_overlay_thickness"] = uhpc_overlay_thickness



# ==========================================================
# MATERIAL COST INPUT
# ==========================================================

st.subheader("Material Unit Cost")


col1, col2 = st.columns(2)


with col1:

    concrete_cost = st.number_input(
        "Concrete Cost ($/m³)",
        value=4000.0,
        step=100.0,
        help="Example: Enter 4000 for $4000/m³"
    )


with col2:

    uhpc_cost = st.number_input(
        "UHPC Cost ($/m³)",
        value=16000.0,
        step=100.0,
        help="Example: Enter 16000 for $16000/m³"
    )


# ==========================================================
# MAINTENANCE INPUT
# ==========================================================

st.subheader("Maintenance Schedule and Unit Cost")


col1, col2 = st.columns(2)


# ---------------- Concrete ----------------

with col1:

    st.markdown("### Concrete Deck")

    concrete_major_interval = st.number_input(
        "Major Rehabilitation Interval (Years)",
        value=25,
        min_value=1,
        step=1,
        help="Example: Major rehabilitation every 25 years"
    )


    concrete_minor_interval = st.number_input(
        "Minor Rehabilitation Interval (Years)",
        value=10,
        min_value=1,
        step=1,
        help="Example: Minor maintenance every 10 years"
    )


    concrete_inspection_interval = st.number_input(
        "Inspection Interval (Years)",
        value=2,
        min_value=1,
        step=1,
        help="Example: Inspection every 2 years"
    )


# ---------------- UHPC ----------------

with col2:

    st.markdown("### UHPC Overlay Deck")

    uhpc_minor_interval = st.number_input(
        "Minor Rehabilitation Interval (Years)",
        value=20,
        min_value=1,
        step=1,
        help="UHPC does not include major rehabilitation."
    )


    uhpc_inspection_interval = st.number_input(
        "Inspection Interval (Years)",
        value=5,
        min_value=1,
        step=1
    )



# ==========================================================
# COST INPUT
# ==========================================================

col1, col2, col3 = st.columns(3)


with col1:

    minor_cost = st.number_input(
        "Minor Rehabilitation Cost ($/m²)",
        value=20.0,
        step=1.0,
        help="Example: 20 means $20/m²"
    )


with col2:

    major_cost = st.number_input(
        "Major Rehabilitation Cost ($/m²)",
        value=500.0,
        step=10.0,
        help="Example: 500 means $500/m²"
    )


with col3:

    inspection_cost = st.number_input(
        "Inspection Cost ($/m²)",
        value=2.0,
        step=0.5,
        help="Example: 2 means $2/m²"
    )


# ==========================================================
# DISCOUNT RATE
# ==========================================================

discount_rate = st.number_input(
    "Discount Rate",
    value=0.025,
    step=0.001,
    format="%.3f",
    help="Enter decimal value. Example: 0.025 represents 2.5%"
)

# ==========================================================
# LCCA CALCULATIONS
# ==========================================================

if st.button(
    "Run Life Cycle Cost Analysis",
    key="lcca_button",
    use_container_width=True
):

    # ------------------------------------------------------
    # Geometry calculations
    # ------------------------------------------------------

    deck_area = deck_length * deck_width

    concrete_volume_full = (
        deck_area * total_thickness
    )

    concrete_volume_uhpc_system = (
        deck_area * concrete_thickness_lcca
    )

    uhpc_volume = (
        deck_area * uhpc_overlay_thickness
    )


    # ------------------------------------------------------
    # Initial construction costs
    # ------------------------------------------------------

    initial_concrete_cost = (
        concrete_volume_full *
        concrete_cost
    )


    initial_uhpc_cost = (
        concrete_volume_uhpc_system * concrete_cost
        +
        uhpc_volume * uhpc_cost
    )


    # ------------------------------------------------------
    # Time vectors
    # ------------------------------------------------------

    time_concrete = np.arange(
        0,
        lcca_concrete_year + 1
    )


    time_uhpc = np.arange(
        0,
        lcca_uhpc_year + 1
    )


    # ------------------------------------------------------
    # Maintenance schedules
    # ------------------------------------------------------

    concrete_minor_years = list(
        range(
            concrete_minor_interval,
            lcca_concrete_year + 1,
            concrete_minor_interval
        )
    )


    concrete_major_years = list(
        range(
            concrete_major_interval,
            lcca_concrete_year + 1,
            concrete_major_interval
        )
    )


    concrete_inspection_years = list(
        range(
            concrete_inspection_interval,
            lcca_concrete_year + 1,
            concrete_inspection_interval
        )
    )


    uhpc_minor_years = list(
        range(
            uhpc_minor_interval,
            lcca_uhpc_year + 1,
            uhpc_minor_interval
        )
    )


    uhpc_inspection_years = list(
        range(
            uhpc_inspection_interval,
            lcca_uhpc_year + 1,
            uhpc_inspection_interval
        )
    )


    # ------------------------------------------------------
    # LCC calculation function
    # ------------------------------------------------------

    def calculate_lcc(
        years,
        initial_cost,
        minor_years,
        inspection_years,
        major_years=None
    ):

        results = []

        for t in years:

            total_cost = initial_cost

            inspection_total = 0
            minor_total = 0
            major_total = 0


            # Inspection

            for year in inspection_years:

                if year <= t:

                    cost = (
                        deck_area *
                        inspection_cost /
                        ((1 + discount_rate) ** year)
                    )

                    inspection_total += cost
                    total_cost += cost



            # Minor maintenance

            for year in minor_years:

                if year <= t:

                    cost = (
                        deck_area *
                        minor_cost /
                        ((1 + discount_rate) ** year)
                    )

                    minor_total += cost
                    total_cost += cost



            # Major maintenance

            if major_years is not None:

                for year in major_years:

                    if year <= t:

                        cost = (
                            deck_area *
                            major_cost /
                            ((1 + discount_rate) ** year)
                        )

                        major_total += cost
                        total_cost += cost



            results.append({

                "Year": t,

                "Total LCC": total_cost,

                "Initial Cost": initial_cost,

                "Inspection": inspection_total,

                "Minor Rehab": minor_total,

                "Major Rehab": major_total

            })


        return pd.DataFrame(results)



    # ------------------------------------------------------
    # Compute LCC
    # ------------------------------------------------------

    df_concrete_lcc = calculate_lcc(
        time_concrete,
        initial_concrete_cost,
        concrete_minor_years,
        concrete_inspection_years,
        concrete_major_years
    )


    df_uhpc_lcc = calculate_lcc(
        time_uhpc,
        initial_uhpc_cost,
        uhpc_minor_years,
        uhpc_inspection_years,
        None
    )


    final_concrete = df_concrete_lcc.iloc[-1]

    final_uhpc = df_uhpc_lcc.iloc[-1]


    st.session_state["df_concrete"] = df_concrete_lcc
    st.session_state["df_uhpc"] = df_uhpc_lcc


    # ======================================================
    # MAINTENANCE TIMELINE
    # ======================================================


    st.subheader(
        "Maintenance Timeline"
    )


    fig, ax = plt.subplots(
        figsize=(12,4)
    )


    y_positions = {

        "Initial Construction":3,

        "Major Rehabilitation":2,

        "Minor Rehabilitation":1,

        "Inspection":0

    }



    # Concrete

    ax.scatter(
        0,
        y_positions["Initial Construction"],
        color="red",
        s=100,
        label="Concrete Deck"
    )


    ax.scatter(
        concrete_major_years,
        [y_positions["Major Rehabilitation"]]
        *len(concrete_major_years),
        color="red",
        s=70
    )


    ax.scatter(
        concrete_minor_years,
        [y_positions["Minor Rehabilitation"]]
        *len(concrete_minor_years),
        color="red",
        s=45
    )


    ax.scatter(
        concrete_inspection_years,
        [y_positions["Inspection"]]
        *len(concrete_inspection_years),
        color="red",
        s=20
    )



    # UHPC

    ax.scatter(
        0,
        y_positions["Initial Construction"],
        color="dodgerblue",
        marker="^",
        s=120,
        label="UHPC Overlay Deck"
    )


    ax.scatter(
        uhpc_minor_years,
        [y_positions["Minor Rehabilitation"]]
        *len(uhpc_minor_years),
        color="dodgerblue",
        marker="^",
        s=70
    )


    ax.scatter(
        uhpc_inspection_years,
        [y_positions["Inspection"]]
        *len(uhpc_inspection_years),
        color="dodgerblue",
        marker="^",
        s=30
    )


    max_service = max(
        lcca_concrete_year,
        lcca_uhpc_year
    )


    for year in np.arange(
        10,
        max_service + 1,
        10
    ):

        ax.axvline(
            year,
            linestyle=":",
            color="black",
            alpha=0.5
        )


    ax.set_xlim(
        0,
        max_service + 5
    )


    ax.set_yticks(
        list(y_positions.values())
    )

    ax.set_yticklabels(
        list(y_positions.keys()),
        fontsize=14,
        fontweight="bold"
    )


    ax.set_xlabel(
        "Time [Years]",
        fontsize=16,
        fontweight="bold"
    )


    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.4
    )


    ax.legend(
        fontsize=12
    )


    st.pyplot(fig)



    # ======================================================
    # COST CURVE + BAR CHART
    # ======================================================


    st.subheader(
        "Life Cycle Cost Comparison"
    )


    col1, col2 = st.columns(2)



    with col1:

        fig, ax = plt.subplots(
            figsize=(8,5)
        )


        ax.plot(
            df_concrete_lcc["Year"],
            df_concrete_lcc["Total LCC"],
            color="red",
            linewidth=3,
            label="Concrete Deck"
        )


        ax.plot(
            df_uhpc_lcc["Year"],
            df_uhpc_lcc["Total LCC"],
            color="dodgerblue",
            linestyle="--",
            linewidth=3,
            label="UHPC Overlay Deck"
        )


        ax.set_xlabel(
            "Service Life [Years]",
            fontsize=14,
            fontweight="bold"
        )


        ax.set_ylabel(
            "Cumulative LCC ($)",
            fontsize=14,
            fontweight="bold"
        )


        ax.legend()

        ax.grid(alpha=0.3)


        st.pyplot(fig)



    with col2:


        labels = [
            "Concrete Deck",
            "UHPC Overlay"
        ]


        initial = [
            final_concrete["Initial Cost"],
            final_uhpc["Initial Cost"]
        ]


        inspection = [
            final_concrete["Inspection"],
            final_uhpc["Inspection"]
        ]


        minor = [
            final_concrete["Minor Rehab"],
            final_uhpc["Minor Rehab"]
        ]


        major = [
            final_concrete["Major Rehab"],
            final_uhpc["Major Rehab"]
        ]


        fig, ax = plt.subplots(
            figsize=(6,4)
        )


        bar_width = 0.4

        x = np.arange(len(labels))


        ax.bar(
            x,
            initial,
            width=bar_width,
            label="Initial",
            color="#4c72b0",
            edgecolor="black"
        )


        ax.bar(
            x,
            inspection,
            width=bar_width,
            bottom=initial,
            label="Inspection",
            color="#8172b3",
            edgecolor="black"
        )


        bottom2 = np.array(initial) + np.array(inspection)


        ax.bar(
            x,
            minor,
            width=bar_width,
            bottom=bottom2,
            label="Minor Rehab",
            color="#55a868",
            edgecolor="black"
        )


        bottom3 = bottom2 + np.array(minor)


        ax.bar(
            x,
            major,
            width=bar_width,
            bottom=bottom3,
            label="Major Rehab",
            color="#c44e52",
            edgecolor="black"
        )


        ax.set_xticks(x)

        ax.set_xticklabels(
            labels,
            fontsize=12,
            fontweight="bold"
        )


        ax.set_ylabel(
            "Cost ($)",
            fontsize=13,
            fontweight="bold"
        )


        ax.legend(
            fontsize=9,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=1
        )


        ax.grid(False)


        for spine in ax.spines.values():

            spine.set_linewidth(1.2)


        plt.tight_layout()


        st.pyplot(fig)



    # ======================================================
    # SUMMARY TABLE
    # ======================================================


    st.subheader(
        "LCCA Summary"
    )


    table_html = f"""

<style>

.lcca-table {{

width:90%;
margin:auto;
border-collapse:collapse;
font-family:Arial;
font-size:18px;

}}


.lcca-table th {{

background-color:DodgerBlue;
color:white;
padding:12px;
border:1px solid #ccc;

}}


.lcca-table td {{

padding:10px;
text-align:center;
border:1px solid #ccc;

}}


.lcca-table tr:nth-child(even) {{

background-color:#f5f5f5;

}}


.parameter {{

font-weight:bold;
text-align:left;

}}

</style>


<table class="lcca-table">

<tr>

<th>Parameter</th>
<th>Concrete Deck</th>
<th>UHPC Overlay Deck</th>

</tr>


<tr>
<td class="parameter">Service Life (Years)</td>
<td>{lcca_concrete_year}</td>
<td>{lcca_uhpc_year}</td>
</tr>


<tr>
<td class="parameter">Initial Cost ($)</td>
<td>{final_concrete["Initial Cost"]:,.2f}</td>
<td>{final_uhpc["Initial Cost"]:,.2f}</td>
</tr>


<tr>
<td class="parameter">Inspection Cost ($)</td>
<td>{final_concrete["Inspection"]:,.2f}</td>
<td>{final_uhpc["Inspection"]:,.2f}</td>
</tr>


<tr>
<td class="parameter">Minor Rehabilitation ($)</td>
<td>{final_concrete["Minor Rehab"]:,.2f}</td>
<td>{final_uhpc["Minor Rehab"]:,.2f}</td>
</tr>


<tr>
<td class="parameter">Major Rehabilitation ($)</td>
<td>{final_concrete["Major Rehab"]:,.2f}</td>
<td>{final_uhpc["Major Rehab"]:,.2f}</td>
</tr>


<tr>
<td class="parameter">Final LCC ($)</td>
<td><b>{final_concrete["Total LCC"]:,.2f}</b></td>
<td><b>{final_uhpc["Total LCC"]:,.2f}</b></td>
</tr>


</table>

"""


    st.markdown(
        table_html,
        unsafe_allow_html=True
    )




# ====================================================================================================================
# STEP 4 - FLEXURAL RELIABILITY ANALYSIS
# ====================================================================================================================

st.markdown(
    """
    <div class="title-box">Step 4: Flexural Reliability Analysis</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="step-title">Step 4: Structural Reliability Analysis</div>',
    unsafe_allow_html=True
)


st.write(
"""
This section evaluates the structural reliability of the **Concrete Deck** and
**UHPC Overlay Deck** systems by considering the uncertainties associated with
material properties, loading conditions, and reinforcement deterioration.

The objective of this analysis is to determine how corrosion-induced steel loss
affects the probability that the bridge deck will not satisfy its required
flexural capacity during its service life.

### Reliability Analysis Concept

Unlike a deterministic analysis that uses only average values, reliability
analysis considers the natural variability of engineering parameters.

The analysis includes uncertainty in:

- Reinforcement steel area due to corrosion deterioration.
- Steel yield strength.
- Concrete compressive strength.
- Applied service loading.

A Monte Carlo simulation approach is used to evaluate the structural response
over time.

### Input Guidance

Please provide the following material and geometric parameters using the correct
engineering units:

- **Top Clear Cover**
  - Unit: **millimeters (mm)**
  - Definition: Distance from the exposed concrete surface to the center of the
    reinforcement bar.
  - Example: Enter `50` for a 50 mm clear cover.

- **Steel Yield Strength**
  - Unit: **Megapascals (MPa)**
  - Definition: Yield strength of reinforcing steel.
  - Example: Enter `414` for Grade 60 reinforcement steel (approximately
    414 MPa).

- **Concrete Compressive Strength**
  - Unit: **Megapascals (MPa)**
  - Definition: Compressive strength of concrete used in the deck.
  - Example: Enter `40` for concrete with approximately 40 MPa compressive
    strength.

### Connection with Previous Steps

The reliability analysis automatically uses the corrosion prediction results
from Step 2.

The calculated steel area loss caused by corrosion is converted into a reduced
reinforcement area, which is then used to evaluate the remaining flexural
capacity of the deck.

The service life used in this analysis follows the exposure period selected in
the reliability assessment and is independent of the LCCA service life selected
in Step 3.

### Reliability Results

The analysis provides:

- **Probability of Failure (Pf)**
  - Represents the likelihood that the structural resistance becomes smaller
    than the applied demand.

- **Reliability Index (β)**
  - Represents the safety margin of the structure.
  - Higher β values indicate better reliability performance.
  - A decreasing β value indicates deterioration of structural performance over
    time.

The reliability results from this step are combined with the Life Cycle Cost
Analysis results in **Step 5: Bridge Deck Value Index (BDVI)** to evaluate the
overall long-term performance and value of each deck system.

### Important Note

For meaningful comparison between the Concrete Deck and UHPC Overlay Deck,
maintain consistent assumptions regarding:

- Reinforcement properties.
- Deck geometry.
- Loading conditions.
- Corrosion prediction inputs.

"""
)


# ==========================================================
# CHECK REQUIRED DATA FROM PREVIOUS STEPS
# ==========================================================


required_variables = [

    "original_steel_diameter",

    "prediction_results_concrete",

    "prediction_results_uhpc",

    "deck_width",

    "concrete_thickness_full"

]


missing=[]


for var in required_variables:

    if var not in st.session_state:

        missing.append(var)



if missing:

    st.error(
        f"""
        Missing information from previous steps:

        {missing}

        Please complete Step 2 and Step 3 before running reliability analysis.
        """
    )


else:


    # ======================================================
    # USER INPUTS
    # ======================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        top_clear_cover = st.number_input(

            "Top Clear Cover (mm)",

            min_value=10.0,

            max_value=150.0,

            value=50.0,

            step=1.0,

            help="Example: 50 means 50 mm cover"

        )


    with col2:

        fy_mean = st.number_input(

            "Steel Yield Strength (MPa)",

            min_value=200.0,

            max_value=800.0,

            value=414.0,

            step=1.0,

            help="Example: 414 MPa"

        )


    with col3:

        fc_mean = st.number_input(

            "Concrete Compressive Strength (MPa)",

            min_value=10.0,

            max_value=150.0,

            value=41.9,

            step=0.1,

            help="Example: 41.9 MPa"

        )


    st.markdown("<br>", unsafe_allow_html=True)



    col1,col2,col3 = st.columns([1,2,1])


    with col2:

        run_reliability = st.button(

            "Run Reliability Analysis",

            key="run_reliability",

            use_container_width=True

        )



    # ======================================================
    # RUN ANALYSIS
    # ======================================================

    if run_reliability:


        import numpy as np

        from scipy.stats import norm, lognorm



        rng = np.random.default_rng(42)



        # --------------------------------------------------
        # Retrieve previous-step information
        # --------------------------------------------------

        original_steel_diameter = (
            st.session_state["original_steel_diameter"]
        )


        deck_width = (
            st.session_state["deck_width"]
        )


        deck_thickness = (
            st.session_state["concrete_thickness_full"]
            *
            1000
        )


        prediction_results_concrete = (
            st.session_state["prediction_results_concrete"]
        )


        prediction_results_uhpc = (
            st.session_state["prediction_results_uhpc"]
        )



        # --------------------------------------------------
        # Reliability parameters
        # --------------------------------------------------

        N_sim = 10000


        steel_cov = 0.10

        load_cov = 0.10

        fc_cov = 0.15

        fy_cov = 0.10



        P_service = 39      # kN


        b = 150             # mm


        L = (
            deck_width *
            1000
        )/2



        As0 = (

            np.pi *
            original_steel_diameter**2 /
            4

        )



        d = (

            deck_thickness
            -
            (
                top_clear_cover
                +
                original_steel_diameter/2
            )

        )



        # ==================================================
        # Corrosion Data
        # ==================================================

        time_concrete = (

            prediction_results_concrete
            ["Time (Years)"]
            .to_numpy()

        )


        rust_concrete = (

            prediction_results_concrete
            ["Rust Expansion (mm)"]
            .to_numpy()

        )



        time_uhpc = (

            prediction_results_uhpc
            ["Time (Years)"]
            .to_numpy()

        )


        rust_uhpc = (

            prediction_results_uhpc
            ["Rust Expansion (mm)"]
            .to_numpy()

        )



        # ==================================================
        # Rust Expansion --> Steel Area Loss
        # ==================================================

        original_area = (

            np.pi *
            original_steel_diameter**2 /
            4

        )



        def area_loss_calculation(rust):

            losses=[]


            for r in rust:


                remaining_diameter = max(

                    original_steel_diameter-r,

                    0

                )


                remaining_area = (

                    np.pi *
                    remaining_diameter**2 /
                    4

                )


                losses.append(

                    original_area -
                    remaining_area

                )


            return np.array(losses)



        loss_concrete = area_loss_calculation(
            rust_concrete
        )


        loss_uhpc = area_loss_calculation(
            rust_uhpc
        )



        # ==================================================
        # Monte Carlo Simulation
        # ==================================================

        def run_MC(time, loss):

            Pf=[]

            beta=[]


            for i in range(len(time)):


                As_mean = max(

                    As0-loss[i],

                    1

                )


                mu_ln = (

                    np.log(As_mean)
                    -
                    0.5*steel_cov**2

                )


                As = lognorm(

                    s=steel_cov,

                    scale=np.exp(mu_ln)

                ).rvs(

                    N_sim,

                    random_state=rng

                )


                fy = rng.normal(

                    fy_mean,

                    fy_cov*fy_mean,

                    N_sim

                )


                fc = rng.normal(

                    fc_mean,

                    fc_cov*fc_mean,

                    N_sim

                )


                fy=np.clip(fy,1,None)

                fc=np.clip(fc,1,None)



                a=(

                    As*fy

                )/(

                    0.85*fc*b

                )



                MR=(

                    As*
                    fy*
                    (d-a/2)

                )



                P=rng.normal(

                    P_service*1000,

                    load_cov*P_service*1000,

                    N_sim

                )



                MS=(

                    P*L/4

                )



                g=MR-MS



                pf=np.mean(g<0)


                pf=np.clip(

                    pf,

                    1e-10,

                    1-1e-10

                )


                Pf.append(pf)


                beta.append(

                    -norm.ppf(pf)

                )


            return np.array(Pf), np.array(beta)



        Pf_c,beta_c = run_MC(

            time_concrete,

            loss_concrete

        )


        Pf_u,beta_u = run_MC(

            time_uhpc,

            loss_uhpc

        )


        st.session_state["beta_c"] = beta_c
        st.session_state["beta_u"] = beta_u

        st.session_state["Pf_c"] = Pf_c
        st.session_state["Pf_u"] = Pf_u



        # ==================================================
        # PLOTS SIDE BY SIDE
        # ==================================================

        col1,col2 = st.columns(2)



        with col1:


            fig,ax = plt.subplots(

                figsize=(8,5)

            )


            ax.plot(

                time_concrete,

                Pf_c,

                'o-',

                color="red",

                label="Concrete Deck"

            )


            ax.plot(

                time_uhpc,

                Pf_u,

                's--',

                color="dodgerblue",

                label="UHPC Overlay"

            )


            ax.set_xlabel(

                "Time [Years]",

                fontsize=15,

                fontweight="bold"

            )


            ax.set_ylabel(

                "Probability of Exceedence ($P_f$)",

                fontsize=15,

                fontweight="bold"

            )


            ax.legend(frameon=False)


            ax.grid(alpha=0.3)


            st.pyplot(fig)



        with col2:


            fig,ax = plt.subplots(

                figsize=(8,5)

            )


            ax.plot(

                time_concrete,

                beta_c,

                'o-',

                color="red",

                label="Concrete Deck"

            )


            ax.plot(

                time_uhpc,

                beta_u,

                's--',

                color="dodgerblue",

                label="UHPC Overlay"

            )


            ax.set_xlabel(

                "Time [Years]",

                fontsize=15,

                fontweight="bold"

            )


            ax.set_ylabel(

                "Reliability Index ($\\beta$)",

                fontsize=15,

                fontweight="bold"

            )


            ax.legend(frameon=False)


            ax.grid(alpha=0.3)


            st.pyplot(fig)





# ====================================================================================================================
# STEP 5
# BRIDGE DECK VALUE INDEX (BDVI)
# ====================================================================================================================

st.markdown(
    """
    <div class="title-box">Step 5: Bridge Deck Value Index (BDVI)</div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="step-title">Step 5: Bridge Deck Value Index (BDVI)</div>',
    unsafe_allow_html=True
)


st.write(
"""
This section combines the results from the **Structural Reliability Analysis**
and the **Life Cycle Cost Analysis (LCCA)** into a single performance indicator
called the **Bridge Deck Value Index (BDVI)**.

The purpose of BDVI is to support decision-making by considering both:

- How well the bridge deck maintains its structural performance over time.
- How economically efficient the bridge deck system is throughout its service
  life.

A deck system with excellent reliability but extremely high cost may not always
be the preferred solution. Similarly, a low-cost option with poor long-term
performance may not provide the desired level of safety. The BDVI provides a
balanced comparison between these two aspects.

### BDVI Weight Selection

Users define the importance of reliability and cost using two weighting factors:

- **Reliability Weight**
  - Represents the importance assigned to structural performance.
  - A higher value means greater priority is given to maintaining safety and
    reliability.

- **Cost Weight**
  - Represents the importance assigned to economic performance.
  - A higher value means greater priority is given to reducing life-cycle cost.

The two weights must satisfy:

**Reliability Weight + Cost Weight = 1.0**

### Weight Selection Examples

The selection of weights depends on the user's design priorities:

- **Safety-focused evaluation**
  - Reliability Weight = `0.80`
  - Cost Weight = `0.20`
  - The analysis gives higher importance to structural performance.

- **Balanced evaluation**
  - Reliability Weight = `0.50`
  - Cost Weight = `0.50`
  - Reliability and economic performance are considered equally important.

- **Cost-focused evaluation**
  - Reliability Weight = `0.30`
  - Cost Weight = `0.70`
  - The analysis gives higher importance to reducing life-cycle cost.

Please enter weights as decimal values.

Examples:

- Enter `0.8`, not `80%`
- Enter `0.2`, not `20%`

### BDVI Calculation Concept

The BDVI combines:

- Normalized reliability performance.
- Normalized life-cycle cost performance.

The calculated BDVI value represents the overall performance value of each
bridge deck alternative relative to the initial condition.

A higher BDVI indicates better overall performance considering the selected
priority between reliability and cost.

### Interpretation of Results

The final BDVI output includes:

- BDVI performance curve over the selected service life.
- Comparison between Concrete Deck and UHPC Overlay Deck.
- Percentage reduction in BDVI from the initial condition.

The percentage reduction indicates how much overall performance is lost over
time:

- Smaller BDVI reduction indicates better long-term performance retention.
- Larger BDVI reduction indicates greater deterioration in combined performance.

### Important Note

The final BDVI result depends on the selected weighting factors. Therefore,
different users may obtain different preferred solutions depending on whether
their priority is:

- Maximum structural reliability.
- Minimum life-cycle cost.
- A balanced combination of both.

For consistent comparison, users should ensure that the reliability and LCCA
inputs represent the same bridge deck alternatives and compatible service-life
assumptions.

"""
)

# ==========================================================
# Weight Inputs
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    w1 = st.number_input(

        "Reliability Weight",

        min_value=0.0,

        max_value=1.0,

        value=0.80,

        step=0.05,

        format="%.2f",

        help="Example: 0.80"
    )


with col2:

    w2 = st.number_input(

        "Life-Cycle Cost Weight",

        min_value=0.0,

        max_value=1.0,

        value=0.20,

        step=0.05,

        format="%.2f",

        help="Example: 0.20"
    )

st.markdown(
    '<div class="step-title">Selected Weighting Factors</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"Reliability Weight = **{w1:.2f}**"
    )


with col2:

    st.success(
        f"Life-Cycle Cost Weight = **{w2:.2f}**"
    )

# ==========================================================
# Check Weight Sum
# ==========================================================

weights_valid = np.isclose(
    w1 + w2,
    1.0,
    atol=1e-6
)

if weights_valid:

    st.success(
        "✅ Weighting factors are valid."
    )

else:

    st.error(
        "The Reliability and Cost weights must add up to 1.00."
    )
# ==========================================================
# Prediction Button
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:

    predict_bdvi = st.button(

        "Calculate Bridge Deck Value Index",

        key="bdvi_prediction",

        use_container_width=True
    )
# ==========================================================
# START BDVI ANALYSIS
# ==========================================================

if predict_bdvi:

    if not weights_valid:

        st.stop()

    # ---------------------------------------------
    # Reliability Data from Step 4
    # ---------------------------------------------

    beta_c = np.asarray(st.session_state["beta_c"])
    beta_u = np.asarray(st.session_state["beta_u"])

    time_concrete = np.arange(
        1,
        len(beta_c)+1
    )

    time_uhpc = np.arange(
        1,
        len(beta_u)+1
    )

    # ---------------------------------------------
    # LCCA Results from Step 3
    # ---------------------------------------------

    df_concrete = st.session_state["df_concrete"].copy()
    df_uhpc = st.session_state["df_uhpc"].copy()
    

    df_concrete_BDVI = (
        df_concrete[
            df_concrete["Year"].isin(
                time_concrete
            )
        ]
        .copy()
    )

    df_uhpc_BDVI = (
        df_uhpc[
            df_uhpc["Year"].isin(
                time_uhpc
            )
        ]
        .copy()
    )

    df_concrete_BDVI = (
        df_concrete_BDVI
        .sort_values("Year")
    )

    df_uhpc_BDVI = (
        df_uhpc_BDVI
        .sort_values("Year")
    )

    LCC_c = (
      df_concrete_BDVI[
        "Total LCC"
       ].to_numpy()
    )

    LCC_u = (
      df_uhpc_BDVI[
        "Total LCC"
      ].to_numpy()
    )





    # ==========================================================
    # NORMALIZATION FUNCTIONS
    # ==========================================================

    def normalize_beta(beta):
        """
        Normalize reliability with respect to
        the initial reliability.
        """
        return beta / beta[0]


    def normalize_cost(cost):
        """
        Normalize cumulative life-cycle cost.
        """
        return cost / np.max(cost)


    # ==========================================================
    # NORMALIZE RELIABILITY
    # ==========================================================

    beta_c_norm = normalize_beta(beta_c)

    beta_u_norm = normalize_beta(beta_u)


    # ==========================================================
    # NORMALIZE COST
    # ==========================================================

    cost_c_norm = normalize_cost(LCC_c)

    cost_u_norm = normalize_cost(LCC_u)


    # ==========================================================
    # BDVI EQUATION
    # ==========================================================

    def calculate_bdvi(beta_norm, cost_norm):

        return (
            w1 * beta_norm
            -
            w2 * cost_norm
        )


    BDVI_c = calculate_bdvi(

        beta_c_norm,

        cost_c_norm

    )


    BDVI_u = calculate_bdvi(

        beta_u_norm,

        cost_u_norm

    )


    # ==========================================================
    # SHIFT FOR VISUAL COMPARISON
    # Concrete starts at 1.0
    # ==========================================================

    shift = 1.0 - BDVI_c[0]

    BDVI_c_plot = BDVI_c + shift

    BDVI_u_plot = BDVI_u + shift


    # ==========================================================
    # CALCULATE PERCENTAGE LOSS
    # ==========================================================

    BDVI_c_initial = BDVI_c_plot[0]

    BDVI_u_initial = BDVI_u_plot[0]


    BDVI_c_drop = (

        (BDVI_c_initial - BDVI_c_plot)

        /

        BDVI_c_initial

        * 100

    )


    BDVI_u_drop = (

        (BDVI_u_initial - BDVI_u_plot)

        /

        BDVI_u_initial

        * 100

    )


    # ==========================================================
    # CREATE RESULT TABLES
    # ==========================================================

    BDVI_concrete = pd.DataFrame({

        "Year": time_concrete,

        "Reliability Index": beta_c,

        "Life-Cycle Cost": LCC_c,

        "BDVI": BDVI_c_plot,

        "BDVI Loss (%)": BDVI_c_drop

    })


    BDVI_uhpc = pd.DataFrame({

        "Year": time_uhpc,

        "Reliability Index": beta_u,

        "Life-Cycle Cost": LCC_u,

        "BDVI": BDVI_u_plot,

        "BDVI Loss (%)": BDVI_u_drop

    })


    # ==========================================================
    # FINAL PERFORMANCE VALUES
    # ==========================================================

    final_concrete_loss = BDVI_c_drop[-1]

    final_uhpc_loss = BDVI_u_drop[-1]


    final_concrete_bdvi = BDVI_c_plot[-1]

    final_uhpc_bdvi = BDVI_u_plot[-1]


    max_service_year = max(

        time_concrete[-1],

        time_uhpc[-1]

    )
        # ==========================================================
    # BDVI COMPARISON PLOT
    # ==========================================================

    st.subheader(
        "Bridge Deck Value Index (BDVI)"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # ----------------------------------------------------------
    # Concrete Deck
    # ----------------------------------------------------------

    ax.plot(

        time_concrete,

        BDVI_c_plot,

        color="red",

        linestyle="-",

        linewidth=2.8,

        marker="o",

        markersize=5,

        label="Concrete Deck"

    )


    # ----------------------------------------------------------
    # UHPC Overlay Deck
    # ----------------------------------------------------------

    ax.plot(

        time_uhpc,

        BDVI_u_plot,

        color="dodgerblue",

        linestyle="--",

        linewidth=2.8,

        marker="s",

        markersize=5,

        label="UHPC Overlay Deck"

    )


    # ----------------------------------------------------------
    # Labels
    # ----------------------------------------------------------

    ax.set_xlabel(

        "Service Life [Years]",

        fontsize=18,

        fontweight="bold"

    )


    ax.set_ylabel(

        "Bridge Deck Value Index (-)",

        fontsize=18,

        fontweight="bold"

    )


    # ----------------------------------------------------------
    # Limits
    # ----------------------------------------------------------

    ax.set_xlim(

        0,

        max_service_year

    )


    # ----------------------------------------------------------
    # Grid
    # ----------------------------------------------------------

    ax.grid(

        alpha=0.20

    )


    # ----------------------------------------------------------
    # Minor ticks
    # ----------------------------------------------------------

    ax.xaxis.set_minor_locator(

        AutoMinorLocator()

    )


    ax.yaxis.set_minor_locator(

        AutoMinorLocator()

    )


    # ----------------------------------------------------------
    # Tick styling
    # ----------------------------------------------------------

    ax.tick_params(

        axis="both",

        which="major",

        labelsize=14,

        width=1.3,

        length=7

    )


    ax.tick_params(

        axis="both",

        which="minor",

        width=1.0,

        length=4

    )


    # ----------------------------------------------------------
    # Border
    # ----------------------------------------------------------

    for spine in ax.spines.values():

        spine.set_linewidth(1.5)


    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

    ax.legend(

        fontsize=14,

        frameon=False,

        loc="best"

    )


    plt.tight_layout()

    st.pyplot(fig)
    # ==========================================================
    # BDVI SUMMARY TABLE
    # ==========================================================

    st.subheader(
        "Bridge Deck Value Index (BDVI) Summary"
    )

    table_html = f"""
    <style>

    .bdvi-table {{
        width:90%;
        margin:auto;
        border-collapse:collapse;
        font-family:Arial;
        font-size:18px;
    }}

    .bdvi-table th {{
        background-color:DodgerBlue;
        color:white;
        padding:12px;
        border:1px solid #ccc;
    }}

    .bdvi-table td {{
        padding:10px;
        text-align:center;
        border:1px solid #ccc;
    }}

    .bdvi-table tr:nth-child(even) {{
        background:#f5f5f5;
    }}

    .parameter {{
        font-weight:bold;
        text-align:left;
    }}

    </style>

    <table class="bdvi-table">

    <tr>
        <th>Parameter</th>
        <th>Concrete Deck</th>
        <th>UHPC Overlay Deck</th>
    </tr>

    <tr>
        <td class="parameter">Service Life (Years)</td>
        <td>{time_concrete[-1]}</td>
        <td>{time_uhpc[-1]}</td>
    </tr>

    <tr>
        <td class="parameter">Reliability Weight</td>
        <td>{w1:.2f}</td>
        <td>{w1:.2f}</td>
    </tr>

    <tr>
        <td class="parameter">Cost Weight</td>
        <td>{w2:.2f}</td>
        <td>{w2:.2f}</td>
    </tr>

    <tr>
        <td class="parameter">Final BDVI</td>
        <td>{BDVI_c_plot[-1]:.3f}</td>
        <td>{BDVI_u_plot[-1]:.3f}</td>
    </tr>

    <tr>
        <td class="parameter">BDVI Reduction (%)</td>
        <td><b>{BDVI_c_drop[-1]:.2f}%</b></td>
        <td><b>{BDVI_u_drop[-1]:.2f}%</b></td>
    </tr>

    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)
  


# ====================================================================================================================
# CLOSING REMARKS / FOOTER
# ====================================================================================================================

st.markdown(
"""
<hr>

<div style="
font-size:16px;
line-height:1.6;
color:#333333;
">

<b>Closing Remarks</b>


This application provides an integrated framework for comparing conventional
concrete bridge decks and UHPC overlay deck systems by combining environmental
exposure, corrosion deterioration, structural reliability, and life-cycle cost
considerations.

The results are intended to support engineering evaluation and preliminary
decision-making by providing a quantitative comparison of long-term performance
and economic value.

The predicted outcomes depend on the assumptions, input parameters, and model
calibration ranges selected by the user. Therefore, the results should be
interpreted as a performance assessment tool rather than a replacement for
project-specific design verification, field investigation, or detailed
structural evaluation.

For meaningful comparisons, users are encouraged to maintain consistent
assumptions regarding geometry, materials, exposure conditions, service life,
and maintenance strategies when evaluating different bridge deck alternatives.

<br>

<b>Developed as an integrated bridge deck performance assessment framework</b>

</div>

<hr>
""",
unsafe_allow_html=True
)



#streamlit run "c:\Users\hu32\Desktop\Ensemble ML\Chloride Content\GUI\Rheo_Streamlit_app.py"

#cd "c:/Users/hu32/Desktop/Ensemble ML/Chloride Content/GUI"
#streamlit run Rheo_Streamlit_app.py
#streamlit run GUI_NJDoT.py
