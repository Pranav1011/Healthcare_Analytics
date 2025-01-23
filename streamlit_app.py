import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
)
st.title("🏥 Healthcare Analytics Dashboard")
 

model_path = 'src/models/ed_wait_time_model.pkl'
encoder_path = 'src/models/encoder.pkl'

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    model_loaded = True
except Exception as e:
    st.error("Error loading the model or encoder. Please ensure the model has been trained and saved.")
    model_loaded = False



@st.cache_data
def load_data():
    admissions = pd.read_csv('data/processed/admissions.csv', parse_dates=['admittime', 'dischtime', 'edregtime', 'edouttime'])
    hourly_stats_admissions = pd.read_csv('data/processed/hourly_stats_admissions.csv')
    hourly_stats_transfers = pd.read_csv('data/processed/hourly_stats_transfers.csv')
    metrics_wait_times = pd.read_csv('data/processed/metrics_wait_times.csv')  
    ward_metrics = pd.read_csv('data/processed/metrics_ward_metrics.csv')     

    return admissions, hourly_stats_admissions, hourly_stats_transfers, metrics_wait_times, ward_metrics

admissions, hourly_admissions, hourly_transfers, wait_times_metrics, ward_metrics = load_data()

st.sidebar.title("⚙️ Dashboard Controls")

# Hourly Admission Patterns
st.header("📊 Hourly Admission Patterns")
admission_type = st.sidebar.selectbox(
    "Select Admission Type",
    options=hourly_admissions['admission_type'].unique(),
    index=0
)
hour_range = st.sidebar.slider(
    "Select Hour Range",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1
)
filtered_admissions = hourly_admissions[
    (hourly_admissions['admission_type'] == admission_type) &
    (hourly_admissions['hour'] >= hour_range[0]) &
    (hourly_admissions['hour'] <= hour_range[1])
]
fig_admissions = px.area(
    filtered_admissions,
    x='hour',
    y='admission_count',
    title=f"Hourly Admissions for {admission_type}",
    labels={'hour': 'Hour of Day', 'admission_count': 'Number of Admissions'},
    color_discrete_sequence=['#1f77b4']
)

st.plotly_chart(fig_admissions, use_container_width=True)

# Admission Type Distribution Pie Chart
st.header("**Admission Type Distribution**")

admission_distribution = hourly_admissions.groupby('admission_type')['admission_count'].sum().reset_index()

fig_pie = px.pie(
    admission_distribution,
    names='admission_type',
    values='admission_count',
    title="Admission Type Distribution",
    color_discrete_sequence=px.colors.sequential.RdBu
)

st.plotly_chart(fig_pie, use_container_width=True)

# Department Utilization
st.header("🏥 Department Utilization Metrics")
selected_ward = st.sidebar.selectbox(
    "Select Ward ID",
    options=ward_metrics['curr_wardid'].unique()  
)
filtered_ward = ward_metrics[ward_metrics['curr_wardid'] == selected_ward]
if not filtered_ward.empty:
    st.write(f"Metrics for Ward ID: {selected_ward}")
    st.metric("Average Length of Stay (hrs)", filtered_ward['length_of_stay_mean'].values[0])
    st.metric("Median Length of Stay (hrs)", filtered_ward['length_of_stay_median'].values[0])
    st.metric("Patient Count", filtered_ward['subject_id_count'].values[0])
else:
    st.warning(f"No data available for Ward ID {selected_ward}")

# Wait Time Analysis
st.header("⏱️ Wait Time Analysis")
wait_type = st.sidebar.selectbox(
    "Select Admission Type for Wait Time Analysis",
    options=wait_times_metrics.index.unique()
)
filtered_wait_time = wait_times_metrics.loc[wait_type]
st.write(f"Wait Time Metrics for {wait_type}")
st.metric("Mean ED Wait Time (min)", filtered_wait_time['ed_wait_time_mean'])
st.metric("Median ED Wait Time (min)", filtered_wait_time['ed_wait_time_median'])

# Prediction Section
st.header("🔮 Predict ED Wait Times")
if model_loaded:

    admission_type_input = st.selectbox("Admission Type", options=['Emergency', 'Elective', 'Urgent'])
    hour_input = st.slider("Hour of Day", 0, 23, value=12)
    admission_location_input = st.text_input("Admission Location", "CLINIC")
    ethnicity_input = st.text_input("Ethnicity", "WHITE")

    input_data = pd.DataFrame({
        'admission_type': [admission_type_input],
        'hour': [hour_input],
        'admission_location': [admission_location_input],
        'ethnicity': [ethnicity_input]
    })


    try:
        input_encoded = encoder.transform(input_data)
        predicted_wait_time = model.predict(input_encoded)

        st.success(f"Predicted ED Wait Time: **{predicted_wait_time[0]:.2f} minutes**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.warning("Model not loaded. Please train and save the model first.")

# Transfer Patterns
st.header("🔄 Transfer Patterns")
transfer_hour_range = st.sidebar.slider(
    "Select Hour Range for Transfers",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1
)
filtered_transfers = hourly_transfers[
    (hourly_transfers['hour'] >= transfer_hour_range[0]) &
    (hourly_transfers['hour'] <= transfer_hour_range[1])
]
fig_transfers = px.bar(
    filtered_transfers,
    x='hour',
    y='transfer_count',
    color='curr_wardid',
    title="Hourly Transfer Patterns",
    labels={'hour': 'Hour of Day', 'transfer_count': 'Transfer Count', 'curr_wardid': 'Ward ID'}
)
st.plotly_chart(fig_transfers, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<b>Created by Sai Pranav</b><br>"
    "Powered by Streamlit | Data Source: MIMIC-III</div>",
    unsafe_allow_html=True
)