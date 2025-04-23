import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import random

# Load the pre-trained model and scaler
try:
    model = load_model("traffic_lstm_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler = joblib.load("scaler.save")
    model_loaded = True
except FileNotFoundError:
    st.error("Error: 'traffic_lstm_model.h5' or 'scaler.save' not found. Please ensure these files are in the same directory.")
    model_loaded = False

# Imaginary locations and route information
locations = ['City Center', 'North Hills', 'East Station', 'West Market', 'South Garden']
route_map = {
    ('City Center', 'North Hills'): ['Route A1', 'Route A2', 'Route A3'],
    ('City Center', 'East Station'): ['Route B1', 'Route B2', 'Route B3'],
    ('North Hills', 'South Garden'): ['Route C1', 'Route C2', 'Route C3'],
    ('East Station', 'West Market'): ['Route D1', 'Route D2', 'Route D3'],
    ('West Market', 'South Garden'): ['Route E1', 'Route E2', 'Route E3']
}

# Traffic threshold for diversion suggestion
TRAFFIC_THRESHOLD = 0.75  # Using percentage (75%)

# Simulate realistic traffic data with daily variations and fake traffic
def generate_realistic_traffic():
    traffic_data = []
    for hour in range(24):
        if 7 <= hour < 9:
            traffic = random.randint(9500, 15000)  # High traffic in the morning
        elif 12 <= hour < 14:
            traffic = random.randint(5000, 8000)  # Moderate traffic during lunch hours
        elif 17 <= hour < 19:
            traffic = random.randint(11000, 17000)  # High traffic in the evening
        else:
            traffic = random.randint(3000, 6000)  # Lower traffic during late night or early morning
        traffic_data.append(traffic)
    return np.array(traffic_data).reshape(-1, 1)

# Fake traffic data for each route
route_fake_traffic = {
    'Route A1': 12500, 'Route A2': 8000, 'Route A3': 9500,
    'Route B1': 11500, 'Route B2': 7000, 'Route B3': 10500,
    'Route C1': 13500, 'Route C2': 7000, 'Route C3': 12000,
    'Route D1': 9000, 'Route D2': 15000, 'Route D3': 19000,
    'Route E1': 11000, 'Route E2': 8500, 'Route E3': 10000
}

# Normalize traffic to percentage (0-100%)
def normalize_traffic_to_percentage(traffic_value, max_traffic=20000):
    return min(100, max(0, (traffic_value / max_traffic) * 100))

# Estimate travel time based on predicted traffic
def estimate_travel_time(route, predicted_traffic):
    # Example: Routes have different distances (in kilometers) and average speeds (in km/h)
    route_distances = {
        'Route A1': 15, 'Route A2': 18, 'Route A3': 20,
        'Route B1': 12, 'Route B2': 16, 'Route B3': 14,
        'Route C1': 25, 'Route C2': 22, 'Route C3': 24,
        'Route D1': 14, 'Route D2': 17, 'Route D3': 19,
        'Route E1': 16, 'Route E2': 20, 'Route E3': 22
    }

    # Average speed in km/h (base value) and traffic speed reduction factor (adjusted based on traffic)
    base_speed = 50  # Base average speed without traffic (in km/h)
    traffic_speed_factor = 1 - (predicted_traffic / 100)  # Reduces speed with more traffic

    # Calculate the travel time (in hours)
    route_distance = route_distances.get(route, 15)  # Default to 15 km if the route isn't found
    adjusted_speed = base_speed * traffic_speed_factor
    travel_time = route_distance / adjusted_speed  # Time in hours

    # Convert to minutes
    return travel_time * 60  # Convert from hours to minutes

# --- Streamlit Interface ---
st.set_page_config(page_title="Smart Road Traffic Predictor", page_icon="🛣️")

# Custom CSS for a black background and white text
st.markdown(
    """
    <style>
        :root {
            --primary-color: #4285F4;
            --secondary-color: #DB4437;
            --tertiary-color: #F4B400;
            --quaternary-color: #0F9D58;
            --background-color: #000000; /* Black background */
            --text-color: #FFFFFF; /* White text */
            --accent-color: #2E2E2E; /* Darker shade for accents */
        }
        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Roboto', sans-serif;
        }
        .stApp {
            background-color: var(--background-color);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
        }
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5em 1em;
            font-size: 1em;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #357AE8;
        }
        .stSelectbox > label {
            color: var(--text-color);
        }
        .stSelectbox div div div {
            background-color: var(--accent-color); /* Darker background for selectbox */
            color: var(--text-color);
            border: 1px solid #333;
            border-radius: 4px;
        }
        .stMarkdown h3 {
            color: var(--secondary-color);
        }
        .stSuccess {
            color: var(--quaternary-color);
            background-color: var(--accent-color);
            padding: 0.8em;
            border-radius: 4px;
        }
        .stWarning {
            color: var(--tertiary-color);
            background-color: var(--accent-color);
            padding: 0.8em;
            border-radius: 4px;
        }
        .stError {
            color: var(--secondary-color);
            background-color: var(--accent-color);
            padding: 0.8em;
            border-radius: 4px;
        }
        .route-card {
            border: 1px solid #333;
            border-radius: 8px;
            padding: 1em;
            margin-bottom: 1em;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            background-color: var(--accent-color); /* Dark background for cards */
            color: var(--text-color);
        }
        .route-card strong {
            color: var(--primary-color);
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.title("🛣️ Smart Road Traffic Predictor")
st.subheader("Find the best route for your journey:")

col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("From:", locations)
with col2:
    destination = st.selectbox("To:", [loc for loc in locations if loc != source])

if st.button("Get Best Route"):
    if not model_loaded:
        st.stop()

    key = (source, destination)
    reverse_key = (destination, source)

    if key in route_map or reverse_key in route_map:
        routes = route_map.get(key, route_map.get(reverse_key))
        predictions = {}

        st.markdown("### 📈 Predicting traffic for available routes...")

        for route in routes:
            # Generate fake traffic data for each route from the pre-defined values
            fake_traffic = route_fake_traffic.get(route, 0)

            # Predict traffic using the LSTM model
            dummy_data = generate_realistic_traffic()
            try:
                scaled_input = scaler.transform(dummy_data).reshape(1, 24, 1)
                predicted_scaled = model.predict(scaled_input)

                # Check if prediction is NaN and handle
                predicted_traffic = scaler.inverse_transform(predicted_scaled)[0][0]
                if np.isnan(predicted_traffic):
                    predicted_traffic = fake_traffic  # Use fake traffic if prediction is invalid
            except Exception as e:
                st.error(f"Error during prediction for {route}: {e}")
                predicted_traffic = fake_traffic # Fallback to fake traffic in case of error

            # Normalize traffic to percentage
            traffic_percentage = normalize_traffic_to_percentage(predicted_traffic)
            predictions[route] = traffic_percentage

            # Estimate travel time for each route based on traffic
            travel_time = estimate_travel_time(route, traffic_percentage)

        st.markdown("---")
        st.markdown("### 🚦 Route Analysis")

        for route in routes:
            traffic_percentage = predictions[route]
            travel_time = estimate_travel_time(route, traffic_percentage)
            with st.container():
                st.markdown(f"<div class='route-card'><strong>Route:</strong> {route}</div>", unsafe_allow_html=True)
                st.write(f"Predicted Traffic: {round(traffic_percentage, 2)}%")
                st.write(f"Estimated Travel Time: {round(travel_time, 2)} minutes")

        st.markdown("---")
        st.markdown("### 🚦 Best Route Recommendation")

        best_route = min(predictions, key=predictions.get)
        best_value = predictions[best_route]

        if best_value > TRAFFIC_THRESHOLD * 100:  # 75% traffic threshold
            alt_routes = [r for r in routes if r != best_route]
            st.warning(f"⚠️ **{best_route}** has high traffic ({round(best_value, 2)}%)! Consider alternative routes: {', '.join(alt_routes)}")
        else:
            st.success(f"✅ Best route: **{best_route}** with predicted traffic: {round(best_value, 2)}%")
            travel_time = estimate_travel_time(best_route, best_value)
            st.write(f"⏳ Estimated Travel Time for the best route: {round(travel_time, 2)} minutes")
    else:
        st.error("No routes found for the selected source and destination.")