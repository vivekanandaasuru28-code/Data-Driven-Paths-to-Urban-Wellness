# ----------------- Professional Urban Planner Dashboard -----------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from sklearn.linear_model import LinearRegression
from geopy.geocoders import Nominatim
from folium.plugins import DualMap, HeatMap
from geopy.distance import great_circle 

# ----------------- Page Setup (MUST BE FIRST STREAMLIT CALL) -----------------
# Initialize session state for modal control and theme
if 'show_settings_modal' not in st.session_state:
    st.session_state.show_settings_modal = False
if 'show_feedback_modal' not in st.session_state:
    st.session_state.show_feedback_modal = False
if 'show_profile_modal' not in st.session_state:
    st.session_state.show_profile_modal = False
    
# CRITICAL FIX 1: Theme Color Initialization (Included for completeness, though not actively used in the provided CSS)
if 'theme_color' not in st.session_state:
    st.session_state.theme_color = '#2F4F4F' 
if 'temp_theme_color' not in st.session_state:
    st.session_state.temp_theme_color = st.session_state.theme_color


st.set_page_config(page_title="Urban Planner Dashboard", layout="wide")

# ----------------- MODAL CONTROL FUNCTIONS -----------------

def open_settings():
    st.session_state.temp_theme_color = st.session_state.theme_color
    st.session_state.show_settings_modal = True
def close_settings():
    st.session_state.show_settings_modal = False

def open_feedback():
    st.session_state.show_feedback_modal = True
    st.session_state.show_profile_modal = False # Ensure only one modal is open
def close_feedback():
    st.session_state.show_feedback_modal = False

def open_profile():
    st.session_state.show_profile_modal = True
    st.session_state.show_feedback_modal = False # Ensure only one modal is open
def close_profile():
    st.session_state.show_profile_modal = False
    
def apply_theme():
    """Update the theme color in session state and rerun the app."""
    if st.session_state.theme_color != st.session_state.temp_theme_color:
        st.session_state.theme_color = st.session_state.temp_theme_color
        st.session_state.show_settings_modal = False 
        st.rerun() 
    else:
        st.session_state.show_settings_modal = False 

# ----------------- CUSTOM CSS AND HEADER LAYOUT -----------------

# CRITICAL FIX 2: Use a raw string (r"...") for the Windows path to avoid SyntaxError
IMAGE_PATH = r"C:\Users\HP\Desktop\app3.jpg"

st.markdown("""
<style>
/* CSS to ensure Streamlit containers/columns display correctly */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 20px;
}

/* Adjustments for the new image logo */
.stImage {
    margin-right: 15px;
    display: flex;
    align-items: center;
}

.titles .title {
    font-weight: bold;
    /* Increased Size */
    font-size: 45px; 
}

.titles .subtitle {
    /* Increased Size */
    font-size: 20px;
    color: #555;
}

.stButton > button {
    height: 38px;
}
</style>
""", unsafe_allow_html=True)

# Use st.columns for the whole row (Hybrid Method)
# Adjusted Ratios for larger logo/title: [Logo, Title/Subtitle, Gap, Feedback Button, Profile Button]
col_logo, col_title, col_gap, col_feedback, col_profile = st.columns([0.7, 4, 1.3, 1, 1])

# --- LOGO SECTION ---
with col_logo:
    try:
        # Increased Image Width to match increased column size
        st.image(IMAGE_PATH, width=150) 
    except FileNotFoundError:
        st.error(f"Logo image not found! Please ensure '{IMAGE_PATH}' is in the correct directory.")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# --- TITLE AND BUTTONS ---
with col_title:
    # Sizes are set in the CSS, but explicit HTML sizes ensure robust rendering
    st.markdown('''
        <div style="margin-top: -5px;">
            <div class="titles title" style="font-weight: bold; font-size: 45px;">ARC OF RENEWAL</div>
            <div class="titles subtitle" style="font-size: 20px; color: #555;">Data-Driven Paths to Urban Wellness</div>
        </div>
    ''', unsafe_allow_html=True)

with col_feedback:
    st.button("💬Feedback", key="btn_feedback", on_click=open_feedback)

with col_profile:
    st.button("👤 Profile", key="btn_profile", on_click=open_profile)

st.markdown("---") 

# ----------------- MODAL CONTAINER LOGIC -----------------

# Feedback Modal
if st.session_state.show_feedback_modal:
    with st.container(height=300, border=True):
        st.subheader("Feedback Form")
        st.write("We value your input. Please share your suggestions or report bugs.")
        
        feedback_text = st.text_area("Your Feedback", height=150, key='modal_feedback_text')
        
        send_col, cancel_col = st.columns([1, 1])
        
        with send_col:
            if st.button("Send Feedback", key='send_feedback_btn'):
                if feedback_text:
                    st.success("Thank you! Your feedback has been successfully sent.")
                    close_feedback()
                else:
                    st.error("Please enter a message before sending.")
        
        with cancel_col:
            st.button("Cancel", key='cancel_feedback_btn', on_click=close_feedback)

# Profile Modal
if st.session_state.show_profile_modal:
    with st.container(height=300, border=True):
        st.subheader("👤 User Profile")
        st.write("Manage your professional profile and account details.")
        
        user_name = st.text_input("Name", "Dr. Jane Doe", key='profile_name')
        user_role = st.text_input("Role", "Senior Urban Planner", key='profile_role')
        
        st.checkbox("Receive Weekly Reports", value=True, key='profile_reports')
        st.checkbox("Enable Dark Mode Preference", value=False, key='profile_dark_mode')
        
        save_col, close_col = st.columns([1, 1])

        with save_col:
            if st.button("Save Profile", key='save_profile_btn'):
                st.success(f"Profile for {user_name} saved successfully!")
                close_profile()
        
        with close_col:
            st.button("Close Profile", key='close_profile_btn', on_click=close_profile)

# ----------------- API KEYS (Placeholders) -----------------
# Not used for actual API calls in this simulated script, but included for completeness
FIRMS_API_KEY = "156a7ecd8b3071727b7d5a88969de8ca"
YOUR_PLANET_API_KEY = "PLAK3ccb7b99e2224b0e807b346566c0b3f7"
OPENWEATHER_API_KEY = "16221f1a37afd5b4793c7d90811662c7"

# ----------------- Helper Functions -----------------
@st.cache_data
def get_coordinates(city_name):
    """Gets latitude and longitude for a city name."""
    geolocator = Nominatim(user_agent="urban_planner")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None, None

# --- API Data Fetching (Simplified/Simulated) ---
@st.cache_data
def fetch_weather(lat, lon):
    """Simulates/fetches weather data."""
    try:
        return {
            "Temperature": np.random.uniform(15, 35),
            "Humidity": np.random.randint(40, 90),
            "Wind Speed": np.random.uniform(1, 15),
            "Weather": "clear sky"
        }
    except:
        return {}

@st.cache_data
def fetch_air_quality(lat, lon):
    """Simulates/fetches air quality data."""
    try:
        aqi = np.random.randint(1, 5)
        aqi_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
        
        return {
            "AQI": aqi,
            "AQI_Text": aqi_map.get(aqi, "N/A"),
            "CO": np.random.uniform(0, 1000),
            "NO2": np.random.uniform(0, 50),
            "O3": np.random.uniform(0, 100),
            "PM2_5": np.random.uniform(0, 50)
        }
    except:
        return {}

@st.cache_data
def fetch_firms_fire_data(lat, lon):
    """Simulates fetching fire data, now providing confidence as a weight."""
    try:
        n_fires = np.random.randint(0, 10)
        fire_data = {
            'latitude': lat + (np.random.rand(n_fires) - 0.5) * 0.05,
            'longitude': lon + (np.random.rand(n_fires) - 0.5) * 0.05,
            'brightness': np.random.uniform(300, 400, n_fires),
            'confidence': np.random.randint(50, 100, n_fires)
        }
        df = pd.DataFrame(fire_data)
        
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def calculate_ndvi_simulated(lat, lon, n_points=30):
    """Simulates real NDVI data points for a heatmap (NASA Data Proxy)."""
    lats = lat + (np.random.rand(n_points) - 0.5) * 0.05
    lons = lon + (np.random.rand(n_points) - 0.5) * 0.05
    ndvi_values = np.clip(np.random.normal(0.45, 0.2, n_points), -0.1, 0.9)
    return pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'ndvi_value': ndvi_values
    })

# --- ML and Prediction ---
def generate_prediction(df, features, target_col):
    """Multivariate Linear Regression for ML Insights (No cache here as data changes based on slider)."""
    features = [f for f in features if f in df.columns]
    if not features or target_col not in df.columns or len(df) < 2:
        return pd.DataFrame({f"Predicted {target_col}": np.zeros(len(df))}), pd.DataFrame({'Feature': [], 'Importance': []})
        
    X = df[features].values
    y = df[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    
    pred = model.predict(X)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(model.coef_)
    }).sort_values(by='Importance', ascending=False)
    
    return pd.DataFrame({f"Predicted {target_col}": pred}), importance

# --- Map Generation ---
def add_esri_map(lat, lon, points=[], zoom=13):
    """Creates a Folium map with Esri Satellite tiles."""
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr='Esri Satellite Imagery',
        overlay=False,
        control=True
    ).add_to(m)
    for p in points:
        icon_type = p.get('icon', 'map-marker')
        color = p.get('color', 'blue')
        
        folium.Marker(
            location=p['loc'], 
            popup=p['popup'],
            icon=folium.Icon(color=color, icon=icon_type, prefix='fa')
        ).add_to(m)
    return m

@st.cache_data
def simulate_urban_points(lat, lon, n_points, point_type, icon, color):
    """Generates simulated points for map visualization."""
    lats = lat + (np.random.rand(n_points) - 0.5) * 0.05
    lons = lon + (np.random.rand(n_points) - 0.5) * 0.05
    points = []
    for i in range(n_points):
        points.append({
            'loc': (lats[i], lons[i]),
            'popup': f"{point_type} Point {i+1}",
            'icon': icon,
            'color': color
        })
    return points

@st.cache_data
def simulate_service_points(lat, lon, service_type, n_points, icon, color, random_seed):
    """Generates simulated service facility points and returns both the list and the raw coordinates."""
    np.random.seed(random_seed) # Use different seed for different services
    lats = lat + (np.random.rand(n_points) - 0.5) * 0.05
    lons = lon + (np.random.rand(n_points) - 0.5) * 0.05
    points = []
    coords = []
    for i in range(n_points):
        coords.append((lats[i], lons[i]))
        points.append({
            'loc': (lats[i], lons[i]),
            'popup': f"{service_type} Location {i+1}",
            'icon': icon,
            'color': color
        })
    return points, coords

@st.cache_data
def calculate_avg_proximity(coords):
    """Calculates the average distance (in km) between all pairs of points using great_circle."""
    if len(coords) < 2:
        return 0.0
    
    total_distance = 0
    pair_count = 0
    
    # Calculate distance for every unique pair
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = great_circle(coords[i], coords[j]).km
            total_distance += distance
            pair_count += 1
            
    if pair_count == 0:
        return 0.0
        
    return total_distance / pair_count

@st.cache_data
def simulate_crowd_risk_points(df, lat, lon):
    """Generates simulated points for the Crowd Safety Index map."""
    n_zones = len(df)
    lats = lat + (np.random.rand(n_zones) - 0.5) * 0.05
    lons = lon + (np.random.rand(n_zones) - 0.5) * 0.05
    points = []
    for i in range(n_zones):
        # Determine color based on CSI
        csi = df.loc[i, 'Crowd Safety Index (CSI)']
        color = 'green'
        if csi > 65:
            color = 'red'
        elif csi > 45:
            color = 'orange'
            
        points.append({
            'loc': (lats[i], lons[i]),
            'popup': f"Zone: {df.loc[i, 'Zone']}<br>CSI: {csi:.2f}<br>Density: {df.loc[i, 'Population Density']}",
            'icon': 'exclamation-triangle' if csi > 65 else 'shield',
            'color': color
        })
    return points

# ----------------- DATA SETUP (Stabilized with Caching) -----------------
@st.cache_data
def get_simulated_data():
    """Generates all simulated dataframes once and caches them."""
    np.random.seed(42) # Ensure consistency

    # Land Use Data 
    df_land = pd.DataFrame(np.random.randint(10,100,size=(10,6)),
                             columns=["Agricultural","Residential","Commercial","Industrial","Recreational", "Impervious Surface"])
    df_land["Zone"] = [f"Zone {i+1}" for i in range(10)]

    # Demographic Data (with SVI components)
    df_demo = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Population Density":np.random.randint(500,5000,10),
        "Age Structure":np.random.randint(10,50,10),
        "Gender Ratio":np.random.randint(40,60,10),
        "Low Income (%)": np.random.uniform(5, 30, 10),
        "No Vehicle Access (%)": np.random.uniform(1, 15, 10)
    })
    df_demo["Social Vulnerability Index (SVI)"] = (df_demo["Low Income (%)"] + df_demo["No Vehicle Access (%)"] + df_demo["Age Structure"] / 5) / 3

    # Economic Data
    df_econ = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Avg Income ($k)":np.random.randint(30,120,10),
        "Unemployment Rate (%)":np.random.uniform(2.0, 15.0, 10),
        "Commercial Vacancy (%)":np.random.uniform(5.0, 30.0, 10),
        "Property Value Index":np.random.uniform(90, 150, 10)
    })

    # Hazard Data
    df_hazards = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Industrial Emissions (Tonnes/yr)": np.random.randint(10, 500, 10),
        "Contaminated Sites": np.random.randint(0, 10, 10),
        "Noise Pollution Index": np.random.uniform(50, 90, 10),
        "Proximity to HZ Plants (km)": np.random.uniform(0.5, 10, 10)
    })

    # Infrastructure Data
    df_infra = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Road Quality Index":np.random.randint(50,100,10),
        "Water Loss (%)":np.random.randint(5,40,10),
        "Power Reliability (%)":np.random.uniform(90.0, 99.9, 10),
        "Broadband Coverage (%)":np.random.uniform(60.0, 100.0, 10)
    })
    
    # Transportation Data
    df_transport = pd.DataFrame({
        "Zone": df_demo["Zone"],
        "Traffic Congestion Index": np.random.uniform(20, 90, 10).round(1),
        "Public Transit Ridership (per day)": np.random.randint(500, 25000, 10),
        "Public Transit Efficiency (%)": np.random.uniform(60, 95, 10).round(1),
        "Road Network Density (km/sq km)": ((100 - df_infra["Road Quality Index"]) * 0.05 + 2).round(2),
        "Average Commute Time (min)": np.random.uniform(15, 60, 10).round(1)
    })


    # Crowd Safety Data 
    df_crowd_safety = pd.DataFrame({
        "Zone": df_demo["Zone"],
        "Population Density": df_demo["Population Density"],
        "Commercial Land (%)": df_land["Commercial"],
        "Recreational Land (%)": df_land["Recreational"],
        "Public Event History (Score 1-10)": np.random.uniform(1, 10, 10),
        "Narrow Exit Points (Count)": np.random.randint(1, 10, 10)
    })

    # Calculate a simulated Crowd Safety Index (CSI)
    df_crowd_safety["Crowd Safety Index (CSI)"] = (
        (df_crowd_safety["Population Density"] / 50) + 
        (df_crowd_safety["Commercial Land (%)"] + df_crowd_safety["Recreational Land (%)"]) / 5 + 
        (df_crowd_safety["Public Event History (Score 1-10)"] * 5) + 
        (df_crowd_safety["Narrow Exit Points (Count)"] * 8)
    )
    df_crowd_safety["Crowd Safety Index (CSI)"] = np.clip(df_crowd_safety["Crowd Safety Index (CSI)"] / df_crowd_safety["Crowd Safety Index (CSI)"].max() * 100, 0, 100)
    
    # Return all dataframes
    return df_land, df_demo, df_econ, df_hazards, df_infra, df_crowd_safety, df_transport

# Load all stable, cached data
df_land, df_demo, df_econ, df_hazards, df_infra, df_crowd_safety, df_transport = get_simulated_data()


# ----------------- Location Input -----------------
location_name = st.text_input("Enter City/Location", "Gajuwaka", key="main_location_input") 
lat, lon = get_coordinates(location_name)
if lat is None:
    st.error("Invalid location. Please try again.")
    st.stop()
st.success(f"Showing data for {location_name} ({lat:.4f}, {lon:.4f})")

# ----------------- Service Point Generation (Cached for map and metrics) -----------------
hospital_points, hospital_coords = simulate_service_points(lat, lon, "Hospital", 5, "h-square", "red", 100)
avg_hospital_proximity = calculate_avg_proximity(hospital_coords)

fire_station_points, fire_station_coords = simulate_service_points(lat, lon, "Fire Station", 4, "fire", "orange", 200)
avg_fire_station_proximity = calculate_avg_proximity(fire_station_coords)

college_points, college_coords = simulate_service_points(lat, lon, "College/School", 7, "graduation-cap", "beige", 300)
avg_college_proximity = calculate_avg_proximity(college_coords)


# ----------------- Tabs -----------------
tabs = st.tabs([
    "Maps", "👥 Demographic","🌍 Land Use", "🚦 Transportation",
    "🌱 Environmental","💹 Economic & Market", "🏗 Infrastructure & Utilities",
    "⚖ Legal & Regulatory", "🤝 Community & Stakeholder", "🌦 Weather",
    "📛 Disaster", "☣ Hazards", "ML Insights", "UVI Score", 
    "🚨 Stampedes"
])

# --------------------------------------------------------------------------------------------------
# ----------------- 1) Maps (5 Graphs/Diagrams) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[0]:
    st.header("Maps & Spatial Visualization")
    df_fires = fetch_firms_fire_data(lat, lon)
    df_ndvi_spatial = calculate_ndvi_simulated(lat, lon)

    # Diagram 1: Service Proximity Metrics
    st.subheader("Service Accessibility Metrics")
    col_prox1, col_prox2, col_prox3 = st.columns(3)
    
    with col_prox1:
        st.metric(label="Hospital Proximity (km)", value=f"{avg_hospital_proximity:.2f} km", delta="Lower is better", delta_color="normal")
    with col_prox2:
        st.metric(label="Fire Station Proximity (km)", value=f"{avg_fire_station_proximity:.2f} km", delta="Target: < 5 km", delta_color="normal")
    with col_prox3:
        st.metric(label="College Proximity (km)", value=f"{avg_college_proximity:.2f} km", delta="Lower is better", delta_color="normal")
        
    st.markdown("---")

    # Diagram 2: DualMap: Satellite vs Hazard (Fire)
    st.subheader("Dual Map: Satellite and Active Fire/Hazard Points")
    dm = DualMap(location=[lat, lon], zoom_start=12, tiles=None) 
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri Basemap", name="Satellite Imagery", overlay=False, control=True).add_to(dm.m1)
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap Base", overlay=False, control=True).add_to(dm.m2)
    folium.LayerControl(collapsed=False).add_to(dm)
    if not df_fires.empty:
        st.info(f"Found {len(df_fires)} active fire points near the area (Simulated/FIRMS Data).")
        for idx, row in df_fires.iterrows():
            folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=5 + row['confidence']/20, color='red', fill=True, fill_opacity=0.8, popup=f"Brightness: {row.get('brightness', 'N/A')}, Confidence: {row.get('confidence', 'N/A')}"
            ).add_to(dm.m2)
    st_folium(dm, width=900, height=500, key="dual_fire_map")
    st.markdown("---")

    # Diagram 3: NDVI Spatial Heatmap 
    st.subheader("🍃 Vegetation Health Spatial Heatmap (NDVI)")
    m_ndvi = folium.Map(location=[lat, lon], zoom_start=12, tiles="cartodbpositron")
    heat_data_ndvi = [[row['latitude'], row['longitude'], row['ndvi_value'] * 5] for index, row in df_ndvi_spatial.iterrows()]
    HeatMap(heat_data_ndvi, radius=40, max_zoom=12, name="NDVI").add_to(m_ndvi)
    folium.Marker([lat, lon], popup="Center Location", icon=folium.Icon(color='black', icon='home')).add_to(m_ndvi)
    st_folium(m_ndvi, width=900, height=500, key="ndvi_heatmap")
    st.markdown("---")

    # Diagram 4: Detailed Urban Feature Map - Fire Stations
    st.subheader("🚒 Detailed Map: Fire Station Locations")
    fire_map = add_esri_map(lat, lon, points=fire_station_points, zoom=14)
    st_folium(fire_map, width=900, height=500, key="feature_map_fire_stations") 
    st.markdown("---")
    
    # Diagram 5: Detailed Urban Feature Map - Hospitals 
    st.subheader("🏥 Detailed Map: Hospitals & Healthcare")
    hospital_map = add_esri_map(lat, lon, points=hospital_points, zoom=14)
    st_folium(hospital_map, width=900, height=500, key="feature_map_hospitals") 
    st.markdown("---")


# --------------------------------------------------------------------------------------------------
# ----------------- 2) Demographics (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[1]:
    st.header("Population Vulnerability & Scenario Modeling") 
    
    growth = st.slider("Population Growth (%)", 0, 50, 10)
    df_demo["Projected Density"] = df_demo["Population Density"] * (1 + growth/100)
    
    st.subheader("Population and Social Vulnerability Data")
    st.dataframe(df_demo[["Zone", "Population Density", "Projected Density", "Low Income (%)", "No Vehicle Access (%)", "Social Vulnerability Index (SVI)"]])
    
    col_demo1, col_demo2 = st.columns(2)
    
    # Graph 1: Projected Population Density
    with col_demo1:
        st.plotly_chart(px.bar(df_demo, x="Zone", y="Projected Density", title="1. Projected Population Density"))
    
    # Graph 2: Social Vulnerability Index
    with col_demo2:
        fig_svi = px.bar(df_demo, x="Zone", y="Social Vulnerability Index (SVI)", title="2. Social Vulnerability Index by Zone", color="Social Vulnerability Index (SVI)", color_continuous_scale=px.colors.sequential.Reds)
        st.plotly_chart(fig_svi)
        
    col_demo3, col_demo4 = st.columns(2)
    
    # Graph 3: Population Distribution Pie
    with col_demo3:
        st.plotly_chart(px.pie(df_demo, names="Zone", values="Projected Density", title="3. Population Distribution"))
        
    # Graph 4: Age Structure Trend
    with col_demo4:
        st.plotly_chart(px.line(df_demo, x="Zone", y="Age Structure", title="4. Age Structure Trend"))
        
    # Graph 5: SVI Gauge
    st.subheader("5. Average Social Vulnerability Index (SVI)")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df_demo["Social Vulnerability Index (SVI)"].mean(),
        title={'text': "Average SVI"},
        gauge={'axis': {'range': [0, df_demo["Social Vulnerability Index (SVI)"].max() * 1.2]}, 'bar': {'color': "darkred"}}
    ))
    st.plotly_chart(fig_gauge)


# --------------------------------------------------------------------------------------------------
# ----------------- 3) Land Use (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[2]:
    st.header("Land Utilization Metrics")
    st.subheader("Land Use and Impervious Surface Area")
    st.dataframe(df_land)
    land_avg = df_land[["Agricultural","Residential","Commercial","Industrial","Recreational", "Impervious Surface"]].mean().reset_index()
    land_avg.columns = ["Land Use Type","Average Area"]
    
    col_land1, col_land2 = st.columns(2)
    
    # Graph 1: Average Land Use Distribution
    with col_land1:
        st.plotly_chart(px.bar(land_avg, x="Land Use Type", y="Average Area", title="Average Land Use Distribution"))
        
    # Graph 2: Impervious Surface Area Trend
    with col_land2:
        fig_impervious = px.line(df_land, x="Zone", y="Impervious Surface", title="2. Impervious Surface Area by Zone (Flood Risk Proxy)", markers=True)
        st.plotly_chart(fig_impervious)
        
    col_land3, col_land4 = st.columns(2)
    
    # Graph 3: Overall Land Use Pie
    with col_land3:
        st.plotly_chart(px.pie(land_avg, names="Land Use Type", values="Average Area", title=" Overall Land Use Distribution"))
        
    # Graph 4: Impervious Surface Gauge
    with col_land4:
        fig_gauge_land = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_land["Impervious Surface"].mean(),
            title={'text': " Average Impervious Surface Area (Units)"},
            gauge={'axis': {'range': [0, df_land["Impervious Surface"].max()]}}
        ))
        st.plotly_chart(fig_gauge_land)
        
    # Graph 5: Impervious Surface vs. Residential Land Scatter
    st.subheader("5. Impervious Surface vs. Residential Land Scatter")
    fig_scatter_land = px.scatter(df_land, x="Residential", y="Impervious Surface", 
                                 size="Commercial", color="Zone", 
                                 title="Residential Land vs. Impervious Surface (Sized by Commercial)",
                                 labels={"Residential": "Residential Land (%)", "Impervious Surface": "Impervious Surface (%)"})
    st.plotly_chart(fig_scatter_land)


# --------------------------------------------------------------------------------------------------
# ----------------- 4) Transportation (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[3]:
    st.header("Transportation Networks & Mobility Analysis")
    
    # Graph 1: Key Mobility Indicators
    st.subheader(" Key Mobility Indicators")
    col_trans1, col_trans2, col_trans3, col_trans4 = st.columns(4)
    
    with col_trans1:
        st.metric(label="Avg Traffic Congestion Index", value=f"{df_transport['Traffic Congestion Index'].mean():.1f}/100", delta_color="inverse")
    with col_trans2:
        st.metric(label="Avg Public Transit Efficiency (%)", value=f"{df_transport['Public Transit Efficiency (%)'].mean():.1f}%")
    with col_trans3:
        st.metric(label="Avg Commute Time (min)", value=f"{df_transport['Average Commute Time (min)'].mean():.1f} min", delta_color="inverse")
    with col_trans4:
        st.metric(label="Avg Road Network Density", value=f"{df_transport['Road Network Density (km/sq km)'].mean():.2f}")
    
    st.markdown("---")
    
    col_chart_trans1, col_chart_trans2 = st.columns(2)
    
    # Graph 2: Traffic Congestion Index
    with col_chart_trans1:
        fig_traffic = px.bar(df_transport, x="Zone", y="Traffic Congestion Index", 
                             title="Traffic Congestion Index by Zone",
                             color="Traffic Congestion Index", 
                             color_continuous_scale=px.colors.sequential.YlOrRd)
        st.plotly_chart(fig_traffic)
        
    # Graph 3: Congestion vs. Road Density
    with col_chart_trans2:
        fig_density = px.scatter(df_transport, 
                                 x="Road Network Density (km/sq km)", 
                                 y="Traffic Congestion Index", 
                                 size="Public Transit Ridership (per day)", 
                                 hover_name="Zone",
                                 title=" Congestion vs. Road Density (Sized by Ridership)")
        st.plotly_chart(fig_density)
        
    # Graph 4: Public Transit Performance Multi-line
    st.subheader(" Public Transit Performance")
    fig_transit = px.line(df_transport, x="Zone", y=["Public Transit Ridership (per day)", "Public Transit Efficiency (%)", "Average Commute Time (min)"],
                          title="Ridership, Efficiency, and Commute Time Trends", markers=True)
    st.plotly_chart(fig_transit)

    # Graph 5: Simulated Commute Mode Share
    st.subheader("Simulated Commute Mode Share")
    mode_data = pd.DataFrame({
        'Mode': ['Private Car', 'Public Transit', 'Cycling/Walking', 'Other'],
        'Share': [45, 30, 15, 10]
    })
    fig_mode_pie = px.pie(mode_data, names='Mode', values='Share', title='Simulated Commute Mode Share')
    st.plotly_chart(fig_mode_pie)


# --------------------------------------------------------------------------------------------------
# ----------------- 5) Environmental (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[4]:
    st.header("Environmental & Sustainability Analytics")
    df_env = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Green Space (%)":df_land["Recreational"] + df_land["Agricultural"],
        "Water Quality Index":np.random.randint(50,100,10),
        "Air Quality Index (AQI)":np.random.randint(1,5,10),
        "Biodiversity Score":np.random.uniform(0.5, 1.0, 10)
    })
    df_env = df_env.merge(df_land[['Zone', 'Impervious Surface']], on='Zone')
    st.dataframe(df_env)
    
    col_env1, col_env2 = st.columns(2)
    
    # Graph 1: Green Space Bar
    with col_env1:
        st.plotly_chart(px.bar(df_env, x="Zone", y="Green Space (%)", title="Green Space % by Zone"))
        
    # Graph 2: AQI Bar
    with col_env2:
        fig_aqi = px.bar(df_env, x="Zone", y="Air Quality Index (AQI)", title=" Air Quality Index by Zone", color="Air Quality Index (AQI)", color_continuous_scale=px.colors.sequential.Inferno_r)
        st.plotly_chart(fig_aqi)
        
    col_env3, col_env4 = st.columns(2)

    # Graph 3: Water Quality Trend
    with col_env3:
        st.plotly_chart(px.line(df_env, x="Zone", y="Water Quality Index", title="Water Quality Trend"))
        
    # Graph 4: Biodiversity Gauge
    with col_env4:
        fig_gauge_env = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_env["Biodiversity Score"].mean() * 100,
            title={'text': "Average Biodiversity Score (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkgreen"}}
        ))
        st.plotly_chart(fig_gauge_env)
        
    # Graph 5: Water Quality vs. Impervious Surface
    st.subheader("Water Quality vs. Impervious Surface Scatter")
    fig_scatter_env = px.scatter(df_env, 
                                 x="Impervious Surface", 
                                 y="Water Quality Index", 
                                 size="Green Space (%)", 
                                 color="Zone", 
                                 title="Water Quality Index vs. Impervious Surface (Lower Impervious = Better Quality)")
    st.plotly_chart(fig_scatter_env)


# --------------------------------------------------------------------------------------------------
# ----------------- 6) Economic & Market (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[5]:
    st.header("Ecomic & Markets Data Chart")
    st.subheader("Key Economic Indicators by Zone")
    st.dataframe(df_econ)
    
    col_econ1, col_econ2 = st.columns(2)
    
    # Graph 1: Average Income Bar
    with col_econ1:
        st.plotly_chart(px.bar(df_econ, x="Zone", y="Avg Income ($k)", title="Average Income by Zone ($k)"))
        
    # Graph 2: Unemployment Rate Bar
    with col_econ2:
        fig_unemp = px.bar(df_econ, x="Zone", y="Unemployment Rate (%)", title="Unemployment Rate by Zone", color="Unemployment Rate (%)", color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_unemp)
        
    col_econ3, col_econ4 = st.columns(2)

    # Graph 3: Property Value Index Trend
    with col_econ3:
        st.plotly_chart(px.line(df_econ, x="Zone", y="Property Value Index", title="Property Value Index Trend"))
        
    # Graph 4: Vacancy vs Property Value Scatter
    with col_econ4:
        st.plotly_chart(px.scatter(df_econ, x="Commercial Vacancy (%)", y="Property Value Index", color="Zone", size="Avg Income ($k)", title="Vacancy vs Property Value"))

    # Graph 5: Average Property Value Index Gauge
    st.subheader("Average Property Value Index")
    fig_gauge_econ = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df_econ["Property Value Index"].mean(),
        title={'text': "Average Property Value Index"},
        gauge={'axis': {'range': [90, 150]}, 'bar': {'color': "darkgoldenrod"}}
    ))
    st.plotly_chart(fig_gauge_econ)


# --------------------------------------------------------------------------------------------------
# ----------------- 7) Infrastructure & Utilities (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[6]:
    st.header("Infrastructure & Utilities Assessment")
    st.subheader("Infrastructure Reliability and Coverage")
    st.dataframe(df_infra)
    
    col_infra1, col_infra2 = st.columns(2)
    
    # Graph 1: Road Quality Index Bar
    with col_infra1:
        st.plotly_chart(px.bar(df_infra, x="Zone", y="Road Quality Index", title="1. Road Quality Index"))
        
    # Graph 2: Water Loss Bar
    with col_infra2:
        fig_water = px.bar(df_infra, x="Zone", y="Water Loss (%)", title="2. Water Loss % by Zone (Inefficiency)", color="Water Loss (%)", color_continuous_scale=px.colors.sequential.Reds)
        st.plotly_chart(fig_water)
        
    col_infra3, col_infra4 = st.columns(2)
    
    # Graph 3: Power Reliability Trend
    with col_infra3:
        st.plotly_chart(px.line(df_infra, x="Zone", y="Power Reliability (%)", title="3. Power Reliability Trend"))
        
    # Graph 4: Broadband Gauge
    with col_infra4:
        fig_gauge_infra = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_infra["Broadband Coverage (%)"].mean(),
            title={'text': "4. Average Broadband Coverage (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}
        ))
        st.plotly_chart(fig_gauge_infra)
        
    # Graph 5: Road Quality vs. Water Loss Scatter
    st.subheader("5. Road Quality vs. Water Loss Scatter")
    fig_scatter_infra = px.scatter(df_infra, x="Road Quality Index", y="Water Loss (%)", 
                                   size="Broadband Coverage (%)", color="Zone", 
                                   title="Road Quality Index vs. Water Loss % (Efficiency Trade-offs)")
    st.plotly_chart(fig_scatter_infra)


# --------------------------------------------------------------------------------------------------
# ----------------- 8) Legal & Regulatory (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[7]:
    st.header("Compliance Monitoring & legal Standards")
    df_legal = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Zoning Compliance Score":np.random.randint(60,100,10),
        "Permit Approval Time (Days)":np.random.randint(10,90,10),
        "Litigation Rate (per 1000 pop)":np.random.uniform(0.1, 1.5, 10),
        "Building Code Violations":np.random.randint(0, 5, 10)
    })
    st.subheader("Regulatory Performance Indicators")
    st.dataframe(df_legal)
    
    col_legal1, col_legal2 = st.columns(2)
    
    # Graph 1: Compliance Bar
    with col_legal1:
        st.plotly_chart(px.bar(df_legal, x="Zone", y="Zoning Compliance Score", title=" Zoning Compliance Score"))
        
    # Graph 2: Permit Time Bar
    with col_legal2:
        fig_permit = px.bar(df_legal, x="Zone", y="Permit Approval Time (Days)", title="Permit Approval Time (Days)", color="Permit Approval Time (Days)", color_continuous_scale=px.colors.sequential.Sunset)
        st.plotly_chart(fig_permit)
        
    col_legal3, col_legal4 = st.columns(2)

    # Graph 3: Litigation Rate Line
    with col_legal3:
        st.plotly_chart(px.line(df_legal, x="Zone", y="Litigation Rate (per 1000 pop)", title="Litigation Rate Trend"))
        
    # Graph 4: Average Compliance Gauge
    with col_legal4:
        fig_gauge_legal = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_legal["Zoning Compliance Score"].mean(),
            title={'text': "Average Compliance Score"},
            gauge={'axis': {'range': [60, 100]}, 'bar': {'color': "darkgreen"}}
        ))
        st.plotly_chart(fig_gauge_legal)

    # Graph 5: Violation Distribution Pie (Simulated)
    st.subheader(" Simulated Violation Distribution")
    violation_data = df_legal.groupby('Zone')['Building Code Violations'].sum().reset_index()
    fig_violation_pie = px.pie(violation_data, names='Zone', values='Building Code Violations', title='Building Code Violations by Zone')
    st.plotly_chart(fig_violation_pie)


# --------------------------------------------------------------------------------------------------
# ----------------- 9) Community & Stakeholder (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[8]:
    st.header("Stakeholder Mapping & Community Involvement")
    df_comm = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Engagement Score":np.random.randint(50,100,10),
        "Social Cohesion Index":np.random.uniform(0.6, 0.9, 10),
        "Complaint Volume (Monthly)":np.random.randint(5, 50, 10),
        "Volunteer Rate (%)":np.random.uniform(2.0, 10.0, 10)
    })
    st.subheader("Community Health Metrics")
    st.dataframe(df_comm)
    
    col_comm1, col_comm2 = st.columns(2)
    
    # Graph 1: Engagement Bar
    with col_comm1:
        st.plotly_chart(px.bar(df_comm, x="Zone", y="Engagement Score", title="Community Engagement Score"))
        
    # Graph 2: Complaint Volume Bar
    with col_comm2:
        fig_complaint = px.bar(df_comm, x="Zone", y="Complaint Volume (Monthly)", title="Monthly Complaint Volume", color="Complaint Volume (Monthly)", color_continuous_scale=px.colors.sequential.amp)
        st.plotly_chart(fig_complaint)
        
    col_comm3, col_comm4 = st.columns(2)

    # Graph 3: Cohesion Line
    with col_comm3:
        st.plotly_chart(px.line(df_comm, x="Zone", y="Social Cohesion Index", title="Social Cohesion Index Trend"))
        
    # Graph 4: Volunteer Gauge
    with col_comm4:
        fig_gauge_comm = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_comm["Volunteer Rate (%)"].mean(),
            title={'text': "4. Average Volunteer Rate (%)"},
            gauge={'axis': {'range': [0, df_comm["Volunteer Rate (%)"].max() * 1.2]}, 'bar': {'color': "purple"}}
        ))
        st.plotly_chart(fig_gauge_comm)
        
    # Graph 5: Engagement vs. Complaint Volume Scatter
    st.subheader("5. Engagement Score vs. Complaint Volume Scatter")
    fig_scatter_comm = px.scatter(df_comm, x="Engagement Score", y="Complaint Volume (Monthly)", 
                                   size="Volunteer Rate (%)", color="Zone", 
                                   title="Engagement vs. Complaints (Higher Engagement should correlate with lower complaints/better reporting)")
    st.plotly_chart(fig_scatter_comm)


# --------------------------------------------------------------------------------------------------
# ----------------- 10) Weather (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[9]:
    st.header("Predictive Climate Analysis")
    
    weather_data = fetch_weather(lat, lon)
    air_quality_data = fetch_air_quality(lat, lon)
    
    # Diagram 1: Current Conditions Metrics
    st.subheader(" Current Conditions Summary")
    if weather_data and air_quality_data:
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        col_w1.metric("Temperature (°C)", f"{weather_data.get('Temperature'):.1f}", "Current")
        col_w2.metric("Humidity (%)", f"{weather_data.get('Humidity')}", "")
        col_w3.metric("Wind Speed (m/s)", f"{weather_data.get('Wind Speed'):.1f}", "")
        col_w4.metric("Weather", weather_data.get('Weather').title(), "")
        st.markdown("---")
        
        # Diagram 2: Air Quality Index Gauge
        st.subheader("Air Quality Index (AQI)")
        fig_aqi_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=air_quality_data.get('AQI', 1),
            title={'text': f"AQI: {air_quality_data.get('AQI_Text', 'N/A')}"},
            gauge={'axis': {'range': [1, 5]}, 
                   'bar': {'color': "darkorange"},
                   'steps': [{'range': [1, 2], 'color': 'green'},
                             {'range': [2, 3], 'color': 'yellow'},
                             {'range': [3, 5], 'color': 'red'}]}
        ))
        st.plotly_chart(fig_aqi_gauge)
        st.markdown("---")

        # Diagram 3: Simulated 7-Day Temperature Forecast
        st.subheader(" Simulated 7-Day Temperature Forecast")
        df_forecast = pd.DataFrame({
            "Day": pd.date_range(start="2024-01-01", periods=7).strftime("%A"),
            "Temperature (°C)": np.random.uniform(weather_data.get('Temperature', 25) - 5, weather_data.get('Temperature', 25) + 5, 7)
        })
        st.plotly_chart(px.line(df_forecast, x="Day", y="Temperature (°C)", title="Simulated Daily Temperature Forecast", markers=True))
        
        # Diagram 4: Detailed Air Pollutants Bar Chart
        st.subheader("Detailed Air Pollutants (µg/m³)")
        df_pollutants = pd.DataFrame({
            'Pollutant': ['CO', 'NO2', 'O3', 'PM2.5'],
            'Concentration': [air_quality_data.get('CO', 0), air_quality_data.get('NO2', 0), air_quality_data.get('O3', 0), air_quality_data.get('PM2_5', 0)]
        })
        fig_pollutants = px.bar(df_pollutants, x='Pollutant', y='Concentration', title='Pollutant Concentration Levels')
        st.plotly_chart(fig_pollutants)

        # Diagram 5: Simulated Rainfall Trend Line
        st.subheader("Simulated Monthly Rainfall Trend")
        df_rainfall = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Rainfall (mm)': np.random.uniform(20, 150, 12).round(1)
        })
        fig_rainfall = px.line(df_rainfall, x='Month', y='Rainfall (mm)', title='Simulated Historical Monthly Rainfall', markers=True, color_discrete_sequence=['blue'])
        st.plotly_chart(fig_rainfall)
        
    else:
        st.error("Could not fetch simulated weather or air quality data.")


# --------------------------------------------------------------------------------------------------
# ----------------- 11) Disaster (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[10]:
    st.header("Disaster Preparedness & Response")
    
    # Diagram 1: Active Hazard Monitoring (Header/Info)
    st.subheader("Active Hazard Monitoring (FIRMS Simulation)")
    if not df_fires.empty:
        st.info(f"The Dual Map in the 'Maps' tab shows {len(df_fires)} active fire points. Data below shows fire hotspots.")
        st.dataframe(df_fires.head())
    else:
        st.warning("No active fire data found in the area for the current period.")

    df_disaster = pd.DataFrame({
        "Zone":[f"Zone {i+1}" for i in range(10)],
        "Flood Risk Index":np.random.randint(1, 10, 10),
        "Seismic Risk Index":np.random.randint(1, 10, 10),
        "Tsunami Vulnerability":np.random.randint(0, 10, 10),
        "Evacuation Capacity Score":np.random.randint(50, 95, 10)
    })
    
    st.dataframe(df_disaster)
    col_d1, col_d2 = st.columns(2)
    
    # Graph 2: Flood Risk Bar
    with col_d1:
        fig_flood = px.bar(df_disaster, x="Zone", y="Flood Risk Index", title="Flood Risk Index", color="Flood Risk Index", color_continuous_scale=px.colors.sequential.Blues)
        st.plotly_chart(fig_flood)
        
    # Graph 3: Seismic Risk Bar
    with col_d2:
        fig_seismic = px.bar(df_disaster, x="Zone", y="Seismic Risk Index", title=" Seismic Risk Index", color="Seismic Risk Index", color_continuous_scale=px.colors.sequential.Oranges)
        st.plotly_chart(fig_seismic)
    
    # Graph 4: Evacuation Gauge
    st.subheader("Average Evacuation Capacity Score")
    fig_gauge_dis = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df_disaster["Evacuation Capacity Score"].mean(),
        title={'text': "Average Evacuation Capacity Score"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ))
    st.plotly_chart(fig_gauge_dis)
    
    # Graph 5: Evacuation Score vs. Flood/Seismic Risk Scatter
    st.subheader("Evacuation Capacity vs. Total Risk Scatter")
    df_disaster['Total Risk'] = df_disaster['Flood Risk Index'] + df_disaster['Seismic Risk Index']
    fig_scatter_dis = px.scatter(df_disaster, x="Total Risk", y="Evacuation Capacity Score", 
                                   size="Tsunami Vulnerability", color="Zone", 
                                   title="Evacuation Capacity vs. Total Hazard Risk (Higher Capacity, Lower Risk is Ideal)")
    st.plotly_chart(fig_scatter_dis)

    
# --------------------------------------------------------------------------------------------------
# ----------------- 12) Hazards -----------------
# --------------------------------------------------------------------------------------------------
with tabs[11]:
    st.header("Hazardous Materials & Pollution Analysis")
    st.subheader("Industrial and Environmental Hazards")
    st.dataframe(df_hazards)
    
    col_h1, col_h2 = st.columns(2)
    
    # Graph 1: Emissions Bar
    with col_h1:
        fig_emissions = px.bar(df_hazards, x="Zone", y="Industrial Emissions (Tonnes/yr)", title=" Industrial Emissions", color="Industrial Emissions (Tonnes/yr)", color_continuous_scale=px.colors.sequential.Greys)
        st.plotly_chart(fig_emissions)
        
    # Graph 2: Noise Bar
    with col_h2:
        fig_noise = px.bar(df_hazards, x="Zone", y="Noise Pollution Index", title=" Noise Pollution Index", color="Noise Pollution Index", color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_noise)
        
    col_h3, col_h4 = st.columns(2)

    # Graph 3: Sites vs Proximity Scatter
    with col_h3:
        st.plotly_chart(px.scatter(df_hazards, x="Contaminated Sites", y="Proximity to HZ Plants (km)", color="Zone", title="Contaminated Sites vs Hazardous Plant Proximity"))

    # Graph 4: Average Noise Index Gauge
    with col_h4:
        fig_gauge_haz = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_hazards["Noise Pollution Index"].mean(),
            title={'text': " Average Noise Pollution Index (Target: < 70)"},
            gauge={'axis': {'range': [50, 90]}, 'bar': {'color': "darkmagenta"}}
        ))
        st.plotly_chart(fig_gauge_haz)
        
    # Graph 5: Contaminated Sites Trend Line
    st.subheader("Contaminated Sites Trend")
    fig_sites_line = px.line(df_hazards, x="Zone", y="Contaminated Sites", title="Contaminated Sites Count by Zone", markers=True, color_discrete_sequence=['brown'])
    st.plotly_chart(fig_sites_line)


# --------------------------------------------------------------------------------------------------
# ----------------- 13) ML Insights (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[12]:
    st.header("Machine Learning & Predictive Insights")
    st.info("Using Multivariate Linear Regression on simulated data for predictive modeling.")
    
    # 1. Predict Property Value Index
    st.subheader("Predictive Model: Property Value Index")
    features_prop = st.multiselect(
        "Select Features to Predict Property Value Index",
        ["Avg Income ($k)", "Unemployment Rate (%)", "Commercial Vacancy (%)", "Population Density", "Road Quality Index"],
        default=["Avg Income ($k)", "Commercial Vacancy (%)", "Road Quality Index"]
    )
    
    df_ml = pd.merge(df_econ, df_demo[["Zone", "Population Density"]], on="Zone")
    df_ml = pd.merge(df_ml, df_infra[["Zone", "Road Quality Index"]], on="Zone")
    
    if features_prop:
        df_pred, importance = generate_prediction(df_ml, features_prop, "Property Value Index")
        df_ml = pd.concat([df_ml, df_pred], axis=1)
        df_ml['Residuals'] = df_ml["Property Value Index"] - df_ml["Predicted Property Value Index"]
        
        # Diagram 1: Model Summary/Metrics (Header)
        st.subheader("1. Model Performance Summary")
        mae = np.mean(np.abs(df_ml['Residuals']))
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} Index Points", delta="Lower is better")
        
        col_ml1, col_ml2 = st.columns(2)
        
        # Graph 2: Feature Importance Bar
        with col_ml1:
            st.plotly_chart(px.bar(importance, x="Feature", y="Importance", title="2. Feature Importance"))
            
        # Graph 3: Actual vs Predicted Scatter
        with col_ml2:
            st.plotly_chart(px.scatter(df_ml, x="Property Value Index", y="Predicted Property Value Index", title="3. Actual vs Predicted Property Value Index", color="Zone"))

        # Graph 4: Actual vs Predicted Trend by Zone
        st.subheader("4. Actual vs Predicted Trend")
        df_trend = df_ml[["Zone", "Property Value Index", "Predicted Property Value Index"]]
        fig_trend = px.line(df_trend, x="Zone", y=["Property Value Index", "Predicted Property Value Index"], title="Actual vs Predicted Property Value by Zone", markers=True)
        st.plotly_chart(fig_trend)
        
        # Graph 5: Residuals Distribution Histogram
        st.subheader("5. Residuals Distribution")
        fig_residuals = px.histogram(df_ml, x="Residuals", title="Distribution of Model Residuals (Error)")
        st.plotly_chart(fig_residuals)
        
    else:
        st.warning("Please select at least one feature.")


# --------------------------------------------------------------------------------------------------
# ----------------- 14) UVI Score (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[13]:
    st.header("Urban Vitality Index (UVI) Score")
    st.info("The UVI Score is a composite index calculated from Land Use, Demographics, Economic, and Infrastructure data.")

    # Merge all relevant dataframes
    df_uvi = df_land.merge(df_demo, on="Zone").merge(df_econ, on="Zone").merge(df_infra, on="Zone")

    # Define UVI calculation weights (simulated)
    df_uvi['Economic Score'] = (df_uvi['Avg Income ($k)'] / 120 + (1 - df_uvi['Commercial Vacancy (%)'] / 30) + df_uvi['Property Value Index'] / 150) / 3 * 100
    df_uvi['Social Score'] = (1 - df_uvi['Social Vulnerability Index (SVI)'] / df_uvi['Social Vulnerability Index (SVI)'].max()) * 100
    df_uvi['Infra Score'] = (df_uvi['Power Reliability (%)'] / 100 + (1 - df_uvi['Water Loss (%)'] / 40) + (1 - df_uvi['Impervious Surface'] / 100)) / 3 * 100
    df_uvi['Urban Vitality Index (UVI)'] = (
        df_uvi['Economic Score'] * 0.4 + df_uvi['Social Score'] * 0.3 + df_uvi['Infra Score'] * 0.3
    )

    st.subheader("Composite Urban Vitality Index (UVI) by Zone")
    df_uvi_display = df_uvi[["Zone", "Urban Vitality Index (UVI)", "Economic Score", "Social Score", "Infra Score"]].sort_values(by="Urban Vitality Index (UVI)", ascending=False)
    st.dataframe(df_uvi_display)
    
    col_uvi1, col_uvi2 = st.columns(2)
    
    # Graph 1: UVI Bar
    with col_uvi1:
        fig_uvi = px.bar(df_uvi_display, x="Zone", y="Urban Vitality Index (UVI)", title="1. Urban Vitality Index (UVI)", color="Urban Vitality Index (UVI)", color_continuous_scale=px.colors.sequential.Mint)
        st.plotly_chart(fig_uvi)
        
    # Graph 2: Component Breakdown Bar
    with col_uvi2:
        fig_comp = px.bar(df_uvi_display, x="Zone", y=["Economic Score", "Social Score", "Infra Score"], title="2. UVI Component Breakdown")
        st.plotly_chart(fig_comp)
        
    col_uvi3, col_uvi4 = st.columns(2)

    # Graph 3: UVI Gauge
    with col_uvi3:
        fig_gauge_uvi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_uvi["Urban Vitality Index (UVI)"].mean(),
            title={'text': "3. Overall UVI Score (Target: 75+)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkcyan"}, 'steps': [
                {'range': [0, 50], 'color': 'red'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': 'green'}
            ]}
        ))
        st.plotly_chart(fig_gauge_uvi)
        
    # Graph 4: Economic vs Social Score Scatter
    with col_uvi4:
        fig_scatter_uvi = px.scatter(df_uvi, x="Economic Score", y="Social Score", 
                                   size="Infra Score", color="Zone", 
                                   title="4. Economic vs Social Score (Sized by Infrastructure)")
        st.plotly_chart(fig_scatter_uvi)

    # Graph 5: UVI Trend Line
    st.subheader("5. Urban Vitality Index Trend")
    fig_line_uvi = px.line(df_uvi_display, x="Zone", y="Urban Vitality Index (UVI)", title="UVI Trend by Zone", markers=True, color_discrete_sequence=['darkcyan'])
    st.plotly_chart(fig_line_uvi)


# --------------------------------------------------------------------------------------------------
# ----------------- 15) Stampedes & Crowd Safety (5 Graphs) -----------------
# --------------------------------------------------------------------------------------------------
with tabs[14]:
    st.header("🚨 Stampedes & Crowd Safety Risk Assessment")
    st.info("Crowd Safety Index (CSI) is calculated based on population density, commercial/recreational land use, and risk factors like exit points.")

    df_crowd_safety_sorted = df_crowd_safety.sort_values(by="Crowd Safety Index (CSI)", ascending=False)
    st.subheader("Crowd Safety Index (CSI) Data")
    st.dataframe(df_crowd_safety_sorted)

    col_csi1, col_csi2 = st.columns(2)
    
    # Graph 1: CSI Bar
    with col_csi1:
        fig_csi = px.bar(df_crowd_safety_sorted, x="Zone", y="Crowd Safety Index (CSI)", title="1. Crowd Safety Index (CSI) by Zone (Higher is Worse)", color="Crowd Safety Index (CSI)", color_continuous_scale=px.colors.sequential.Reds)
        st.plotly_chart(fig_csi)
    
    # Diagram 2: CSI Spatial Map
    with col_csi2:
        st.subheader("2. CSI Spatial Risk Map")
        points_csi = simulate_crowd_risk_points(df_crowd_safety_sorted, lat, lon)
        map_csi = add_esri_map(lat, lon, points=points_csi, zoom=13)
        st_folium(map_csi, width=450, height=500, key="csi_risk_map")

    col_csi3, col_csi4 = st.columns(2)

    # Graph 3: Density vs CSI Scatter
    with col_csi3:
        fig_scatter_csi = px.scatter(df_crowd_safety_sorted, x="Population Density", y="Crowd Safety Index (CSI)", 
                                   size="Narrow Exit Points (Count)", color="Zone", 
                                   title="3. Density vs. CSI (Sized by Exit Point Count)")
        st.plotly_chart(fig_scatter_csi)
        
    # Graph 4: Public Event History Trend
    with col_csi4:
        fig_event_line = px.line(df_crowd_safety_sorted, x="Zone", y="Public Event History (Score 1-10)", title="4. Public Event History Score Trend", markers=True, color_discrete_sequence=['darkred'])
        st.plotly_chart(fig_event_line)

    # Graph 5: Average CSI Gauge
    st.subheader("5. Average Crowd Safety Index (CSI)")
    fig_gauge_csi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df_crowd_safety["Crowd Safety Index (CSI)"].mean(),
        title={'text': "Average CSI (Target: < 40)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "maroon"}, 'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40, 65], 'color': 'orange'},
            {'range': [65, 100], 'color': 'red'}
        ]}
    ))
    st.plotly_chart(fig_gauge_csi)
    
    st.markdown("---")
    st.subheader("Crowd Risk Mitigation Recommendations")
    st.markdown("""
    Based on the data, the following urban planning steps should be prioritized to control stampedes and crushes:
        
        1.  **Design for Flow (Physical Planning):** Review zones with high "Narrow Exit Points (Count)" to mandate widening of corridors, stairwells, and doorways in high-traffic commercial or recreational areas.
        2.  **Operational Zoning and Management:** Enforce strict, real-time crowd capacity monitoring, especially in zones with high "Population Density".
        3.  **Emergency Preparedness (Communication/Signage):** Mandate high-visibility, international-standard emergency signage for all main gathering points.
    """)

# ----------------- Final Line -----------------
st.success("✅ Urban Planner Dashboard fully loaded, stabilized, and updated with 5 diagrams per tab for a professional presentation!")