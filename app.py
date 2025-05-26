import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessing components
@st.cache_resource
def load_model_components():
    """Load all model components with caching for better performance"""
    try:
        model = joblib.load('energy_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_building = joblib.load('le_building.pkl')
        le_day = joblib.load('le_day.pkl')
        
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
            
        return model, scaler, le_building, le_day, model_info
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None, None

# Load components
model, scaler, le_building, le_day, model_info = load_model_components()

def prepare_features(building_type, square_footage, num_occupants, 
                    appliances_used, avg_temperature, day_of_week):
    """Prepare features for prediction"""
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Building Type': [building_type],
        'Square Footage': [square_footage],
        'Number of Occupants': [num_occupants],
        'Appliances Used': [appliances_used],
        'Average Temperature': [avg_temperature],
        'Day of Week': [day_of_week]
    })
    
    # Encode categorical variables
    input_data['Building Type Encoded'] = le_building.transform(input_data['Building Type'])
    input_data['Day of Week Encoded'] = le_day.transform(input_data['Day of Week'])
    
    # Create additional features
    input_data['Occupancy_Density'] = input_data['Number of Occupants'] / input_data['Square Footage']
    input_data['Appliances_per_Occupant'] = input_data['Appliances Used'] / (input_data['Number of Occupants'] + 1)
    
    # Size category
    if square_footage <= 15000:
        size_cat = 0  # Small
    elif square_footage <= 30000:
        size_cat = 1  # Medium
    else:
        size_cat = 2  # Large
    
    input_data['Size_Category_Encoded'] = size_cat
    
    # Select and order features
    feature_columns = ['Square Footage', 'Number of Occupants', 'Appliances Used', 
                      'Average Temperature', 'Building Type Encoded', 'Day of Week Encoded',
                      'Occupancy_Density', 'Appliances_per_Occupant', 'Size_Category_Encoded']
    
    X = input_data[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def predict_energy_consumption(building_type, square_footage, num_occupants, 
                             appliances_used, avg_temperature, day_of_week):
    """Make energy consumption prediction"""
    try:
        X_scaled = prepare_features(building_type, square_footage, num_occupants, 
                                  appliances_used, avg_temperature, day_of_week)
        prediction = model.predict(X_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main app interface
def main():
    # Header
    st.title("🏢 Energy Consumption Predictor")
    st.markdown("### Predict building energy consumption based on characteristics")
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        if model_info:
            st.metric("Model Accuracy (R²)", f"{model_info['r2_score']:.3f}")
            st.metric("RMSE", f"{model_info['rmse']:.2f}")
            st.metric("MAE", f"{model_info['mae']:.2f}")
            st.info(f"Model: {model_info['model_name']}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔧 Building Characteristics")
        
        # Input fields
        building_type = st.selectbox(
            "Building Type",
            options=['Residential', 'Commercial', 'Industrial'],
            help="Select the type of building"
        )
        
        square_footage = st.number_input(
            "Square Footage",
            min_value=500,
            max_value=100000,
            value=2500,
            step=100,
            help="Total building area in square feet"
        )
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            num_occupants = st.number_input(
                "Number of Occupants",
                min_value=1,
                max_value=500,
                value=10,
                help="Average number of people in the building"
            )
            
            appliances_used = st.number_input(
                "Number of Appliances",
                min_value=1,
                max_value=100,
                value=15,
                help="Total number of electrical appliances"
            )
        
        with col1_2:
            avg_temperature = st.slider(
                "Average Temperature (°C)",
                min_value=-10,
                max_value=40,
                value=20,
                help="Average ambient temperature"
            )
            
            day_of_week = st.selectbox(
                "Day Type",
                options=['Weekday', 'Weekend'],
                help="Type of day for prediction"
            )
    
    with col2:
        st.header("📈 Prediction Results")
        
        if st.button("🔮 Predict Energy Consumption", type="primary"):
            with st.spinner("Calculating prediction..."):
                prediction = predict_energy_consumption(
                    building_type, square_footage, num_occupants,
                    appliances_used, avg_temperature, day_of_week
                )
                
                if prediction is not None:
                    # Display prediction
                    st.success("Prediction Complete!")
                    st.metric(
                        "Predicted Energy Consumption",
                        f"{prediction:.2f} units",
                        help="Energy consumption in standardized units"
                    )
                    
                    # Calculate efficiency metrics
                    energy_per_sqft = prediction / square_footage
                    energy_per_occupant = prediction / num_occupants
                    
                    st.subheader("📊 Efficiency Metrics")
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("Energy per Sq Ft", f"{energy_per_sqft:.4f}")
                    
                    with col2_2:
                        st.metric("Energy per Occupant", f"{energy_per_occupant:.2f}")
                    
                    # Visualization
                    st.subheader("📈 Comparison Chart")
                    
                    # Create comparison data
                    building_averages = {
                        'Residential': 2500,
                        'Commercial': 4200,
                        'Industrial': 6800
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Your Building', f'Average {building_type}'],
                            y=[prediction, building_averages[building_type]],
                            marker_color=['#FF6B6B', '#4ECDC4']
                        )
                    ])
                    
                    fig.update_layout(
                        title="Energy Consumption Comparison",
                        yaxis_title="Energy Consumption",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Example predictions
        st.subheader("💡 Example Predictions")
        examples = [
            ("Small Residential", "Residential", 1500, 3, 10, 22, "Weekday"),
            ("Large Commercial", "Commercial", 25000, 100, 50, 20, "Weekday"),
            ("Industrial Plant", "Industrial", 50000, 200, 80, 18, "Weekend")
        ]
        
        for name, b_type, sq_ft, occupants, appliances, temp, day in examples:
            if st.button(f"Try {name}", key=name):
                pred = predict_energy_consumption(b_type, sq_ft, occupants, appliances, temp, day)
                if pred:
                    st.info(f"{name}: {pred:.2f} units")

    # Footer
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on historical data and should be used as estimates.")

if __name__ == "__main__":
    main()
