from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import requests
from geopy.geocoders import Nominatim

# Load column structure
X_encoded_columns = pd.read_csv("X_encoded_columns.csv")["column_name"].tolist()

# Define the model class
class DiseasePredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiseasePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

# Load the model and scaler globally
num_features = len(X_encoded_columns)
num_classes = len(pd.read_csv("disease.csv")["Predicted Disease"].unique())
model = DiseasePredictionModel(num_features, num_classes)
model.load_state_dict(torch.load("disease_prediction_model.pth", weights_only=True))
model.eval()
scaler = joblib.load("scaler.pkl")

# Define dropdown options
genders = ['Male', 'Female', 'Others']
yes_no = ['Yes', 'No']
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
symptoms = ['Cough', 'Fever', 'Headache', 'Fatigue', 'Nausea']  # Adjust based on training data
chronic_conditions = ['None', 'Asthma', 'Hypertension', 'Diabetes']  # Adjust based on training data

# Initialize Flask app
app = Flask(__name__)

# Helper functions (unchanged from your code)
def get_lat_long(location_name):
    try:
        geolocator = Nominatim(user_agent="disease_predictor")
        location = geolocator.geocode(location_name)
        return (location.latitude, location.longitude) if location else None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

def get_nearby_hospitals_osm(latitude, longitude, radius=5000):
    url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{latitude},{longitude});
      way["amenity"="hospital"](around:{radius},{latitude},{longitude});
      relation["amenity"="hospital"](around:{radius},{latitude},{longitude});
    );
    out center;
    """
    try:
        response = requests.post(url, data={"data": overpass_query})
        if response.status_code == 200:
            hospitals = []
            data = response.json()
            for element in data['elements']:
                if element['type'] == 'node':
                    hospitals.append({
                        "name": element.get('tags', {}).get('name', 'N/A'),
                        "lat": element.get('lat', 'N/A'),
                        "lon": element.get('lon', 'N/A'),
                    })
                elif element['type'] in ['way', 'relation']:
                    hospitals.append({
                        "name": element.get('tags', {}).get('name', 'N/A'),
                        "lat": element['center'].get('lat', 'N/A'),
                        "lon": element['center'].get('lon', 'N/A'),
                    })
            return hospitals
        else:
            print(f"Overpass API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error querying Overpass API: {e}")
        return []

def get_location_data(location):
    api_key = "33097de9e6f4716a711d7a38fb8fcbf6"  # Replace with your actual key
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    try:
        response = requests.get(weather_url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp'] - 273.15
            humidity = data['main']['humidity']
            aqi = 0  # Default since AQI isn’t available in free tier
            return aqi, temp, humidity
        else:
            print(f"Error fetching weather data: {response.status_code}")
            return None, None, None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None, None, None

# Function to predict disease and find hospitals
def predict_disease(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['Location', 'Gender', 'Smoking', 'Alcohol',
                                                 'Chronic Condition', 'Pollen Allergy', 'Season',
                                                 'Symptom 1', 'Symptom 2', 'Symptom 3'])
    missing_cols = [col for col in X_encoded_columns if col not in input_df.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
        input_df = pd.concat([input_df, missing_df], axis=1)
    input_df = input_df[X_encoded_columns]
    continuous_features = ['Age', 'BMI', 'AQI', 'Temp (°C)', 'Humidity (%)']
    input_df[continuous_features] = scaler.transform(input_df[continuous_features])
    input_tensor = torch.tensor(input_df.to_numpy(dtype=np.float32))
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, axis=1).item()
    disease_labels = pd.read_csv("disease.csv")["Predicted Disease"].unique()
    predicted_disease = disease_labels[predicted_class]
    coordinates = get_lat_long(input_data['Location'])
    nearby_hospitals = get_nearby_hospitals_osm(*coordinates) if coordinates else []
    return predicted_disease, nearby_hospitals

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        location = request.form['location']
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        gender = request.form['gender']
        smoking = request.form['smoking']
        alcohol = request.form['alcohol']
        chronic_condition = request.form['chronic_condition']
        pollen_allergy = request.form['pollen_allergy']
        season = request.form['season']
        symptom1 = request.form['symptom1']
        symptom2 = request.form['symptom2']
        symptom3 = request.form['symptom3']

        # Calculate BMI
        bmi = weight / (height ** 2)

        # Fetch environmental data
        aqi, temp, humidity = get_location_data(location)
        if aqi is None:
            aqi, temp, humidity = 0, 25.0, 50.0

        # Prepare input data
        input_data = {
            'Age': age,
            'BMI': bmi,
            'AQI': aqi,
            'Temp (°C)': temp,
            'Humidity (%)': humidity,
            'Location': location,
            'Gender': gender,
            'Smoking': smoking,
            'Alcohol': alcohol,
            'Chronic Condition': chronic_condition,
            'Pollen Allergy': pollen_allergy,
            'Season': season,
            'Symptom 1': symptom1,
            'Symptom 2': symptom2,
            'Symptom 3': symptom3
        }

        # Predict and get results
        predicted_disease, nearby_hospitals = predict_disease(input_data)

        # Render template with results
        return render_template('index.html', prediction=predicted_disease, hospitals=nearby_hospitals,
                               genders=genders, yes_no=yes_no, seasons=seasons,
                               symptoms=symptoms, chronic_conditions=chronic_conditions)
    else:
        # Render form on GET request
        return render_template('index.html', genders=genders, yes_no=yes_no, seasons=seasons,
                               symptoms=symptoms, chronic_conditions=chronic_conditions)

if __name__ == '__main__':
    app.run(debug=True)