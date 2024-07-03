Data Collection

two types natural and technical
and hotspot areas

    Historical Data: Collect historical data on past disasters (earthquakes, floods, hurricanes, wildfires, etc.). Sources include government agencies, meteorological departments, and research institutions.
    Real-time Data: Integrate real-time data feeds such as satellite imagery, weather reports, seismic data, social media feeds, and sensor networks.

2. Data Preprocessing

    Cleaning: Handle missing values, remove duplicates, and filter out noise.
    Normalization: Scale features to ensure uniformity and improve model performance.
    Feature Engineering: Extract relevant features like weather patterns, geographical features, population density, etc.

3. Model Selection and Training

    Machine Learning Models: Consider models like Random Forest, Gradient Boosting Machines (GBMs), or Support Vector Machines (SVMs) for structured data.
    Deep Learning Models: Use Convolutional Neural Networks (CNNs) for image data, Recurrent Neural Networks (RNNs) or Long Short-Term Memory networks (LSTMs) for time-series data.
    Hybrid Models: Combine different models to leverage the strengths of each (e.g., using CNNs for feature extraction from images and LSTMs for temporal patterns).

4. Model Evaluation

    Metrics: Use metrics such as accuracy, precision, recall, F1 score, and Area Under the Curve (AUC) to evaluate model performance.
    Cross-Validation: Implement cross-validation to ensure model robustness and avoid overfitting.

5. Deployment

    Real-time Monitoring: Set up a pipeline to continuously feed real-time data into the model.
    Alert System: Develop an alert system to notify relevant authorities and stakeholders when a potential disaster is detected.
    Response Strategies: Integrate response strategies, such as evacuation plans, resource allocation, and communication protocols, based on model predictions.

6. Continuous Improvement

    Feedback Loop: Incorporate feedback from actual events to improve model accuracy and response effectiveness.
    Adaptation: Continuously update the model with new data and evolving patterns to maintain its predictive power.

Tools and Technologies

    Data Collection and Preprocessing: Python (pandas, numpy), Apache Kafka for real-time data streaming.
    Model Training: TensorFlow, PyTorch, scikit-learn.
    Deployment: Docker, Kubernetes, cloud platforms like AWS, GCP, or Azure.
    Visualization and Alerts: Dash, Flask for web apps, Twilio for SMS alerts.

Would you like more details on any specific part of this approach?
from where to get disaster data
ChatGPT

To build an effective disaster prediction model, you can source disaster data from various reputable sources. Here are some key sources for obtaining historical and real-time disaster data:
Historical Disaster Data

    EM-DAT (International Disaster Database):
        Maintained by the Centre for Research on the Epidemiology of Disasters (CRED).
        Provides detailed information on the occurrence and impacts of over 22,000 mass disasters globally from 1900 to the present.
        EM-DAT

    US Geological Survey (USGS):
        Offers extensive data on earthquakes, volcanoes, and other natural hazards.
        USGS Earthquake Hazards Program

    National Oceanic and Atmospheric Administration (NOAA):
        Provides data on weather-related disasters, including hurricanes, tornadoes, and floods.
        NOAA National Centers for Environmental Information

    European Space Agency (ESA):
        Offers satellite imagery and data on various natural disasters.
        ESA Earth Online

    World Bank Open Data:
        Contains various datasets on disaster risk management and the impacts of natural disasters.
        World Bank Open Data

Real-time Data

    NASA Earth Observing System Data and Information System (EOSDIS):
        Provides real-time satellite data on natural disasters.
        NASA EOSDIS

    National Weather Service (NWS):
        Offers real-time weather data and forecasts.
        NWS

    Global Disaster Alert and Coordination System (GDACS):
        Provides real-time alerts about natural disasters around the world.
        GDACS

    Copernicus Emergency Management Service (CEMS):
        Offers real-time data and maps for emergency management.
        Copernicus EMS

    Social Media Feeds:
        Platforms like Twitter can provide real-time updates on disaster events. APIs can be used to gather and analyze data.

Additional Resources

    OpenStreetMap (OSM): Provides geospatial data that can be used for disaster mapping and analysis.
        OpenStreetMap
    Google Earth Engine: Offers extensive satellite imagery and geospatial datasets for environmental monitoring.
        Google Earth Engine

Temporal Features

    Start Year: Start Year
    Start Month: Start Month
    Start Day: Start Day
    End Year: End Year
    End Month: End Month
    End Day: End Day

Geospatial Features

    Latitude: Latitude
    Longitude: Longitude

Disaster-Specific Features

    Disaster Group: Disaster Group
    Disaster Subgroup: Disaster Subgroup
    Disaster Type: Disaster Type
    Disaster Subtype: Disaster Subtype
    Event Name: Event Name (may need encoding or transformation)

Severity and Impact Features

    Total Deaths: Total Deaths
    No. Injured: No. Injured
    No. Affected: No. Affected
    No. Homeless: No. Homeless
    Total Affected: Total Affected

Other Potential Features

    Country: Country (if considering multiple countries)
    Subregion: Subregion
    Region: Region
    Magnitude: Magnitude
    Magnitude Scale: Magnitude Scale



Predicting whether a disaster will occur and identifying its type involves building a machine learning model that can analyze various features in a dataset. Here are the steps to achieve this:
1. Data Collection

Collect a comprehensive dataset that includes historical records of natural and technical disasters along with various features such as:

    Date and time
    Location (latitude, longitude)
    Weather conditions (temperature, humidity, wind speed, etc.)
    Geological data (seismic activity, fault lines)
    Environmental data (land use, vegetation cover)
    Technical data (infrastructure condition, industrial activity)
    Socioeconomic data (population density, economic indicators)

2. Data Preprocessing

    Cleaning: Handle missing values, remove duplicates, and correct errors.
    Normalization: Normalize numerical features to ensure uniformity.
    Encoding: Convert categorical features into numerical representations using techniques like one-hot encoding or label encoding.
    Feature Engineering: Create new features that might be helpful for prediction, such as interaction terms or temporal features (e.g., season, day of the week).

3. Exploratory Data Analysis (EDA)

    Visualization: Use plots and charts to understand the distribution of features and their relationships with the target variable (disaster occurrence and type).
    Correlation Analysis: Identify correlations between features and the target variable.
    Feature Importance: Use techniques like mutual information or tree-based models to identify important features.

4. Model Selection

Choose appropriate machine learning models. Some options include:

    Logistic Regression: For binary classification (disaster occurrence).
    Random Forest: For handling both classification and regression tasks.
    Gradient Boosting Machines: Such as XGBoost or LightGBM for improved performance on structured data.
    Neural Networks: Especially useful if there are complex, non-linear relationships in the data.

5. Model Training

    Split the Data: Divide the data into training, validation, and test sets.
    Training: Train the model on the training set, tuning hyperparameters using cross-validation.
    Evaluation: Evaluate the model on the validation set using metrics like accuracy, precision, recall, F1-score, and ROC-AUC for classification tasks.

6. Predicting Disaster Type

    Multi-class Classification: Use models like Random Forest, Gradient Boosting, or Neural Networks to predict the type of disaster (e.g., earthquake, flood, industrial accident).
    Multi-label Classification: If disasters can have multiple labels (e.g., an earthquake causing a tsunami), use models that can handle multi-label classification.

7. Model Deployment

    Real-time Predictions: Integrate the model into a system that can provide real-time predictions based on incoming data.
    Monitoring and Maintenance: Continuously monitor model performance and update it with new data to maintain accuracy and relevance.

Example Workflow

python

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('disaster_data.csv')

# Data preprocessing
# Handle missing values
data.fillna(method='ffill', inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['temperature', 'humidity', 'wind_speed', 'seismic_activity']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Encode categorical features
encoder = OneHotEncoder()
categorical_features = ['location', 'season']
encoded_features = encoder.fit_transform(data[categorical_features]).toarray()
data = pd.concat([data, pd.DataFrame(encoded_features)], axis=1)
data.drop(categorical_features, axis=1, inplace=True)

# Split the data
X = data.drop(['disaster_occurrence', 'disaster_type'], axis=1)
y_occurrence = data['disaster_occurrence']
y_type = data['disaster_type']

X_train, X_test, y_train_occurrence, y_test_occurrence = train_test_split(X, y_occurrence, test_size=0.2, random_state=42)
X_train, X_test, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)

# Train the model for disaster occurrence
occurrence_model = RandomForestClassifier()
occurrence_model.fit(X_train, y_train_occurrence)

# Train the model for disaster type
type_model = RandomForestClassifier()
type_model.fit(X_train, y_train_type)

# Evaluate the models
y_pred_occurrence = occurrence_model.predict(X_test)
y_pred_type = type_model.predict(X_test)

print("Disaster Occurrence Model Evaluation:")
print(classification_report(y_test_occurrence, y_pred_occurrence))

print("Disaster Type Model Evaluation:")
print(classification_report(y_test_type, y_pred_type))

This workflow provides a structured approach to predicting disaster occurrence and type using machine learning techniques. Adjustments may be needed based on specific dataset characteristics and requirements.
will providing location and city will predict whether disaster will occur or not
ChatGPT

Providing location and city information can certainly help predict whether a disaster will occur, but it is often insufficient on its own. The prediction of disasters typically requires a combination of many features to capture the complex factors that influence such events. Here's how location and city information can be used in conjunction with other data:
1. Geospatial Features

Location (latitude and longitude) and city can provide geospatial context, which is critical for predicting certain types of disasters:

    Natural Disasters: Earthquakes, floods, hurricanes, etc., are often location-specific. For example, certain areas are more prone to earthquakes due to tectonic activity.
    Technical Disasters: Industrial accidents, infrastructure failures, etc., may be more likely in areas with specific industrial activities or aging infrastructure.

2. Supplementary Data

To enhance the predictive power, additional features should be included:

    Weather Data: Temperature, humidity, wind speed, and precipitation can help predict weather-related disasters.
    Geological Data: Seismic activity, fault lines, and soil composition are crucial for predicting earthquakes and landslides.
    Environmental Data: Information about nearby water bodies, forest cover, and land use can inform flood and wildfire predictions.
    Socioeconomic Data: Population density, building density, and economic indicators can influence the likelihood and impact of disasters.

3. Temporal Data

Including time-related features can also improve predictions:

    Seasonality: Certain disasters are seasonal (e.g., hurricanes are more likely in specific months).
    Time of Day: Some technical disasters might be more likely during certain times due to operational schedules.

Example Workflow with Location and Additional Features

      # Import necessary libraries
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler, OneHotEncoder
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.metrics import classification_report
      
      # Load the dataset
      data = pd.read_csv('disaster_data.csv')
      
      # Data preprocessing
      # Handle missing values
      data.fillna(method='ffill', inplace=True)
      
      # Normalize numerical features
      scaler = StandardScaler()
      numerical_features = ['temperature', 'humidity', 'wind_speed', 'seismic_activity']
      data[numerical_features] = scaler.fit_transform(data[numerical_features])
      
      # Encode categorical features
      encoder = OneHotEncoder()
      categorical_features = ['location', 'season']
      encoded_features = encoder.fit_transform(data[categorical_features]).toarray()
      data = pd.concat([data, pd.DataFrame(encoded_features)], axis=1)
      data.drop(categorical_features, axis=1, inplace=True)
      
      # Split the data
      X = data.drop(['disaster_occurrence', 'disaster_type'], axis=1)
      y_occurrence = data['disaster_occurrence']
      y_type = data['disaster_type']
      
      X_train, X_test, y_train_occurrence, y_test_occurrence = train_test_split(X, y_occurrence, test_size=0.2, random_state=42)
      X_train, X_test, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
      
      # Train the model for disaster occurrence
      occurrence_model = RandomForestClassifier()
      occurrence_model.fit(X_train, y_train_occurrence)
      
      # Train the model for disaster type
      type_model = RandomForestClassifier()
      type_model.fit(X_train, y_train_type)
      
      # Evaluate the models
      y_pred_occurrence = occurrence_model.predict(X_test)
      y_pred_type = type_model.predict(X_test)
      
      print("Disaster Occurrence Model Evaluation:")
      print(classification_report(y_test_occurrence, y_pred_occurrence))
      
      print("Disaster Type Model Evaluation:")
      print(classification_report(y_test_type, y_pred_type))

In this example, location and city are encoded along with other relevant features to improve the model's ability to predict disasters. Including a broad range of features will enhance the model's accuracy and robustness in predicting whether a disaster will occur and identifying its type.
what are all the required information for prediction give a sample dataset
ChatGPT

To create a robust dataset for predicting disasters, you need to gather a variety of features that capture the multifaceted nature of disaster occurrence. Below is a list of essential features, categorized into different groups, followed by a sample dataset.
Essential Features for Disaster Prediction

    Location and Geospatial Data
        Latitude
        Longitude
        City
        Region
        Elevation

    Temporal Data
        Date
        Time
        Season
        Day of the week

    Weather Data
        Temperature
        Humidity
        Wind speed
        Precipitation
        Atmospheric pressure

    Geological Data
        Seismic activity (recent earthquake magnitude and frequency)
        Proximity to fault lines
        Soil composition

    Environmental Data
        Proximity to water bodies (rivers, lakes, oceans)
        Land use (urban, rural, forest, agricultural)
        Vegetation cover

    Socioeconomic Data
        Population density
        Building density
        Infrastructure condition (age and type of buildings)
        Economic indicators (GDP, income levels)

    Technical and Industrial Data
        Industrial activity (type and intensity)
        Infrastructure types (bridges, dams, power plants)
        Historical incidents (previous accidents, maintenance records)

    Disaster Occurrence and Type
        Disaster occurrence (binary: 0 for no disaster, 1 for disaster)
        Disaster type (e.g., earthquake, flood, hurricane, industrial accident)

Sample Dataset  example me india ke 2-3 data lga dege

Below is a simplified example of what the dataset might look like:
Date	Time	Latitude	Longitude	City	Region	Elevation	Temperature	Humidity	Wind Speed	Precipitation	Seismic Activity	Proximity to Fault Line	Soil Composition	Proximity to Water Body	Land Use	Vegetation Cover	Population Density	Building Density	Infrastructure Condition	Industrial Activity	Historical Incidents	Disaster Occurrence	Disaster Type
2024-01-01	08:00	34.0522	-118.2437	Los Angeles	California	89	20.0°C	50%	5 km/h	0 mm	4.2	15 km	Clay	10 km	Urban	Low	3000	High	Poor	High	2	0	None
2024-03-15	14:30	40.7128	-74.0060	New York	New York	10	12.0°C	70%	10 km/h	5 mm	2.0	200 km	Sandy	1 km	Urban	Medium	10000	Very High	Good	Medium	5	1	Flood
2024-05-20	11:00	35.6895	139.6917	Tokyo	Kanto	40	22.0°C	65%	8 km/h	10 mm	5.0	5 km	Rocky	5 km	Urban	High	14000	Very High	Excellent	Very High	3	1	Earthquake
2024-07-04	17:45	51.5074	-0.1278	London	England	35	18.0°C	55%	7 km/h	2 mm	1.0	100 km	Silt	0.5 km	Urban	Low	9000	High	Good	Medium	1	0	None
2024-10-10	06:30	-33.8688	151.2093	Sydney	New South Wales	58	15.0°C	60%	6 km/h	0 mm	3.5	20 km	Clay	2 km	Urban	Medium	5000	Medium	Fair	Low	0	1	Industrial
Explanation

    Date and Time: When the data was recorded.
    Latitude and Longitude: Geospatial coordinates of the location.
    City and Region: Additional location information.
    Elevation: Height above sea level.
    Temperature, Humidity, Wind Speed, Precipitation: Weather-related features.
    Seismic Activity: Recent seismic activity in the area.
    Proximity to Fault Line: Distance to the nearest fault line.
    Soil Composition: Type of soil in the area.
    Proximity to Water Body: Distance to the nearest water body.
    Land Use and Vegetation Cover: Information about land usage and vegetation.
    Population Density and Building Density: Socioeconomic indicators.
    Infrastructure Condition: Quality of infrastructure.
    Industrial Activity: Level of industrial activity.
    Historical Incidents: Number of previous incidents.
    Disaster Occurrence and Type: Target variables indicating if a disaster occurred and its type.

This dataset is illustrative and can be expanded with more features or adjusted to suit specific needs for predicting disasters.
after prediction what kind of response system should be made
ChatGPT

After predicting the occurrence and type of a disaster, it's essential to have a comprehensive response system in place. This response system should aim to minimize the impact on human life, property, and the environment. Here are the key components of an effective disaster response system:
1. Early Warning System

An automated system to alert relevant authorities and the public as soon as a disaster is predicted.

    Alerts and Notifications: Use multiple channels (SMS, email, app notifications, sirens) to disseminate warnings.
    Public Information: Provide clear instructions on evacuation routes, shelter locations, and safety measures.

2. Emergency Communication System

A robust communication infrastructure to coordinate response efforts among various stakeholders.

    Command Center: Establish a central command center to oversee and coordinate the response.
    Communication Network: Ensure reliable communication between first responders, government agencies, NGOs, and the public.
    Real-Time Updates: Use social media, websites, and news outlets to provide real-time updates.

3. Resource Management System

Efficient allocation and deployment of resources to affected areas.

    Inventory Tracking: Maintain an up-to-date inventory of emergency supplies (food, water, medical supplies).
    Logistics Coordination: Use GIS and other tools to optimize the distribution of resources and personnel.
    Mutual Aid Agreements: Pre-arrange mutual aid agreements with neighboring regions for additional support.

4. Emergency Response Teams

Mobilize trained personnel to respond to the disaster.

    Search and Rescue: Deploy teams equipped to handle search and rescue operations.
    Medical Response: Establish field hospitals and deploy medical personnel to provide immediate care.
    Fire and Hazardous Materials Response: Deploy specialized teams for firefighting and handling hazardous materials.

5. Evacuation and Shelter Management

Organize the safe evacuation of people and provide temporary shelters.

    Evacuation Plans: Develop and disseminate evacuation plans with designated routes and transportation options.
    Shelter Locations: Identify and prepare shelters, ensuring they are stocked with essential supplies.
    Registration and Tracking: Implement a system to register and track evacuees to ensure everyone is accounted for.

6. Post-Disaster Recovery

Support the recovery and rebuilding efforts after the immediate response phase.

    Damage Assessment: Conduct assessments to determine the extent of damage and prioritize recovery efforts.
    Financial Assistance: Provide financial aid to affected individuals and businesses.
    Reconstruction: Plan and execute rebuilding projects, focusing on resilience and mitigation to prevent future disasters.

7. Public Education and Training

Prepare the community to respond effectively to disasters.

    Training Programs: Conduct regular training for first responders and the general public.
    Drills and Exercises: Organize drills to test the response system and identify areas for improvement.
    Awareness Campaigns: Run campaigns to educate the public about disaster preparedness and safety measures.

Example Workflow for an Integrated Disaster Response System

    Prediction Alert:
        The system predicts an earthquake in a specific region.
        Automated alerts are sent to authorities and the public via SMS, app notifications, and sirens.

    Emergency Communication:
        The command center activates, coordinating with local emergency services, NGOs, and government agencies.
        Real-time updates are provided through social media, websites, and news outlets.

    Resource Deployment:
        Emergency supplies are dispatched from nearby warehouses.
        Search and rescue teams are mobilized and directed to the affected areas.

    Evacuation and Shelter:
        Evacuation routes are activated, and transportation is arranged for those without vehicles.
        Shelters are opened and staffed, with supplies and medical personnel on-site.

    Medical Response:
        Field hospitals are set up near the disaster area.
        Medical teams provide first aid and trauma care to the injured.

    Recovery Operations:
        Damage assessment teams survey the affected area.
        Financial assistance programs are activated to support rebuilding efforts.

    Public Education:
        Community meetings and training sessions are held to prepare for future disasters.
        Information on building resilient infrastructure and emergency preparedness is disseminated.

Example Response System Workflow Diagram

    Prediction Phase:
        Data Collection -> Prediction Model -> Alert Generation

    Alert and Notification Phase:
        Alert System -> Public and Authorities -> Communication Channels

    Response Coordination Phase:
        Command Center -> Emergency Services -> Communication Network

    Resource Management Phase:
        Resource Inventory -> Logistics Coordination -> Mutual Aid

    Evacuation and Shelter Phase:
        Evacuation Plans -> Transportation -> Shelter Management

    Medical Response Phase:
        Medical Teams -> Field Hospitals -> Trauma Care

    Post-Disaster Recovery Phase:
        Damage Assessment -> Financial Assistance -> Reconstruction

    Public Education and Training Phase:
        Training Programs -> Drills and Exercises -> Awareness Campaigns

By integrating these components into a cohesive disaster response system, communities can better prepare for, respond to, and recover from disasters, thereby minimizing the impact on lives and property.

Let's develop a prediction code using the provided dataset and create a simple interface for making predictions. We'll use Python with pandas for data handling, scikit-learn for machine learning, and streamlit for the web interface.

First, we'll read the dataset and preprocess it. Then, we'll train a machine learning model to predict disaster occurrence and type. Finally, we'll create a streamlit interface for users to input new data and get predictions.
Step 1: Load and Preprocess the Data

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    import joblib
    
    # Load the dataset
    file_path = '/mnt/data/public_emdat_custom_request_2024-07-03_093beef0-a745-4060-867e-56c5b2c9238a.xlsx'
    data = pd.read_excel(file_path)
    
    # Display the first few rows of the dataset
    print(data.head())
    
    # Data preprocessing
    # Identify numerical and categorical features
    numerical_features = ['Latitude', 'Longitude', 'Elevation', 'Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Seismic Activity', 'Proximity to Fault Line', 'Population Density', 'Building Density']
    categorical_features = ['City', 'Region', 'Soil Composition', 'Proximity to Water Body', 'Land Use', 'Vegetation Cover', 'Infrastructure Condition', 'Industrial Activity']
    
    # Define target variables
    target_occurrence = 'Disaster Occurrence'
    target_type = 'Disaster Type'
    
    # Split the data
    X = data[numerical_features + categorical_features]
    y_occurrence = data[target_occurrence]
    y_type = data[target_type]
    
    # Preprocessing pipelines for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define the model
    model_occurrence = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    model_type = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train_occurrence, y_test_occurrence = train_test_split(X, y_occurrence, test_size=0.2, random_state=42)
    X_train, X_test, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
    
    # Train the models
    model_occurrence.fit(X_train, y_train_occurrence)
    model_type.fit(X_train, y_train_type)
    
    # Save the models
    joblib.dump(model_occurrence, 'model_occurrence.pkl')
    joblib.dump(model_type, 'model_type.pkl')
    
    Step 2: Create a Streamlit Interface
    
    Create a new Python script for the Streamlit app:
    
    python
    
    import streamlit as st
    import pandas as pd
    import joblib
    
    # Load the trained models
    model_occurrence = joblib.load('model_occurrence.pkl')
    model_type = joblib.load('model_type.pkl')
    
    st.title('Disaster Prediction System')
    
    # Input fields for user data
    st.header('Enter the details:')
    latitude = st.number_input('Latitude')
    longitude = st.number_input('Longitude')
    city = st.text_input('City')
    region = st.text_input('Region')
    elevation = st.number_input('Elevation')
    temperature = st.number_input('Temperature')
    humidity = st.number_input('Humidity')
    wind_speed = st.number_input('Wind Speed')
    precipitation = st.number_input('Precipitation')
    seismic_activity = st.number_input('Seismic Activity')
    proximity_to_fault_line = st.number_input('Proximity to Fault Line')
    soil_composition = st.selectbox('Soil Composition', options=['Clay', 'Sandy', 'Rocky', 'Silt'])
    proximity_to_water_body = st.number_input('Proximity to Water Body')
    land_use = st.selectbox('Land Use', options=['Urban', 'Rural', 'Forest', 'Agricultural'])
    vegetation_cover = st.selectbox('Vegetation Cover', options=['Low', 'Medium', 'High'])
    population_density = st.number_input('Population Density')
    building_density = st.selectbox('Building Density', options=['Low', 'Medium', 'High', 'Very High'])
    infrastructure_condition = st.selectbox('Infrastructure Condition', options=['Poor', 'Fair', 'Good', 'Excellent'])
    industrial_activity = st.selectbox('Industrial Activity', options=['Low', 'Medium', 'High', 'Very High'])
    
    # Create a dataframe for the input data
    input_data = pd.DataFrame({
        'Latitude': [latitude],
        'Longitude': [longitude],
        'City': [city],
        'Region': [region],
        'Elevation': [elevation],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Precipitation': [precipitation],
        'Seismic Activity': [seismic_activity],
        'Proximity to Fault Line': [proximity_to_fault_line],
        'Soil Composition': [soil_composition],
        'Proximity to Water Body': [proximity_to_water_body],
        'Land Use': [land_use],
        'Vegetation Cover': [vegetation_cover],
        'Population Density': [population_density],
        'Building Density': [building_density],
        'Infrastructure Condition': [infrastructure_condition],
        'Industrial Activity': [industrial_activity]
    })
    
    if st.button('Predict'):
        # Predict disaster occurrence
        disaster_occurrence_prediction = model_occurrence.predict(input_data)
    
    if disaster_occurrence_prediction == 1:
        st.subheader('Disaster Predicted!')
        
        # Predict disaster type
        disaster_type_prediction = model_type.predict(input_data)
        st.write('Disaster Type:', disaster_type_prediction[0])
    else:
        st.subheader('No Disaster Predicted')

Instructions

  Save the first script as train_model.py and run it to train and save the models.
  Save the second script as app.py.
  Run the Streamlit app with the command:


    streamlit run app.py

This setup will allow users to input data and get predictions on whether a disaster will occur and, if so, what type of disaster it will be. Adjust the input fields and options based on your specific dataset features.

