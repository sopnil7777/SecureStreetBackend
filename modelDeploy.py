
from flask import Flask, request, jsonify
import pandas as pd
import requests
import numpy as np
from datetime import datetime

app = Flask(__name__)

neighbor_df=pd.read_parquet('neighborhoodData.parquet')
neighbor_df=neighbor_df.set_index(neighbor_df.columns[0])

def map_weather_description(openweather_description):
    weather_mapping = {
        'Thunderstorm': 'Stormy',
        'Drizzle': 'Rainy',
        'Rain': 'Rainy',
        'Snow': 'Snowy',
        'Mist': 'Foggy',
        'Smoke': 'Foggy',
        'Haze': 'Foggy',
        'Dust': 'Foggy',
        'Fog': 'Foggy',
        'Sand': 'Foggy',
        'Ash': 'Foggy',
        'Squall': 'Stormy',
        'Clear': 'Clear',
        'Clouds': 'Cloudy'
    }
    
    return weather_mapping.get(openweather_description, 'Unknown')

api_key = 'Weather_API'
city_name = 'Toronto'
def get_weather_by_city(api_key, city_name):
    # OpenWeatherMap API endpoint for current weather using city name
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric'

    # Make a GET request to fetch the data
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        # Extract relevant weather data
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        visibility = data['visibility'] / 1000  # Convert from meters to kilometers
        weather_description = data['weather'][0]['main']


        # Return the extracted data
        return {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'visibility': visibility,
            'weather_description': weather_description
        }
    else:
        print("Error fetching weather data:", data.get("message", "Unknown error"))
        return None
    

def getWeek():
    # Get the current time and day
    now = datetime.now()
    current_hour = now.hour
    current_weekday = now.weekday()  # Monday = 0, Sunday = 6
    
    # Determine if it's a weekend
    is_weekend = 1 if current_weekday in [5, 6] else 0  # 1 if Saturday or Sunday, else 0
    
    # Compute the sine and cosine of the day of the week
    DOW_sin = np.sin(2 * np.pi * current_weekday / 7)
    DOW_cos = np.cos(2 * np.pi * current_weekday / 7)
    return is_weekend,DOW_sin,DOW_cos

# Function to categorize time into OCC_TIME_RANGE
def categorize_time(hour):
    if pd.isna(hour):
        return np.nan
    elif 21 <= hour < 24 or 0 <= hour < 4:
        return 'Night'
    elif 4 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Evening'

# Get the OCC_TIME_RANGE
now=datetime.now()
current_hour = now.hour


neighborhoodList=['NEIGHBOURHOOD_158_Agincourt North',
 'NEIGHBOURHOOD_158_Agincourt South-Malvern West',
 'NEIGHBOURHOOD_158_Alderwood',
 'NEIGHBOURHOOD_158_Annex',
 'NEIGHBOURHOOD_158_Avondale',
 'NEIGHBOURHOOD_158_Banbury-Don Mills',
 'NEIGHBOURHOOD_158_Bathurst Manor',
 'NEIGHBOURHOOD_158_Bay-Cloverhill',
 'NEIGHBOURHOOD_158_Bayview Village',
 'NEIGHBOURHOOD_158_Bayview Woods-Steeles',
 'NEIGHBOURHOOD_158_Bedford Park-Nortown',
 'NEIGHBOURHOOD_158_Beechborough-Greenbrook',
 'NEIGHBOURHOOD_158_Bendale South',
 'NEIGHBOURHOOD_158_Bendale-Glen Andrew',
 'NEIGHBOURHOOD_158_Birchcliffe-Cliffside',
 'NEIGHBOURHOOD_158_Black Creek',
 'NEIGHBOURHOOD_158_Blake-Jones',
 'NEIGHBOURHOOD_158_Briar Hill-Belgravia',
 'NEIGHBOURHOOD_158_Bridle Path-Sunnybrook-York Mills',
 'NEIGHBOURHOOD_158_Broadview North',
 'NEIGHBOURHOOD_158_Brookhaven-Amesbury',
 'NEIGHBOURHOOD_158_Cabbagetown-South St.James Town',
 'NEIGHBOURHOOD_158_Caledonia-Fairbank',
 'NEIGHBOURHOOD_158_Casa Loma',
 'NEIGHBOURHOOD_158_Centennial Scarborough',
 'NEIGHBOURHOOD_158_Church-Wellesley',
 'NEIGHBOURHOOD_158_Clairlea-Birchmount',
 'NEIGHBOURHOOD_158_Clanton Park',
 'NEIGHBOURHOOD_158_Cliffcrest',
 'NEIGHBOURHOOD_158_Corso Italia-Davenport',
 'NEIGHBOURHOOD_158_Danforth',
 'NEIGHBOURHOOD_158_Danforth East York',
 'NEIGHBOURHOOD_158_Don Valley Village',
 'NEIGHBOURHOOD_158_Dorset Park',
 'NEIGHBOURHOOD_158_Dovercourt Village',
 'NEIGHBOURHOOD_158_Downsview',
 'NEIGHBOURHOOD_158_Downtown Yonge East',
 'NEIGHBOURHOOD_158_Dufferin Grove',
 'NEIGHBOURHOOD_158_East End-Danforth',
 "NEIGHBOURHOOD_158_East L'Amoreaux",
 'NEIGHBOURHOOD_158_East Willowdale',
 'NEIGHBOURHOOD_158_Edenbridge-Humber Valley',
 'NEIGHBOURHOOD_158_Eglinton East',
 'NEIGHBOURHOOD_158_Elms-Old Rexdale',
 'NEIGHBOURHOOD_158_Englemount-Lawrence',
 'NEIGHBOURHOOD_158_Eringate-Centennial-West Deane',
 'NEIGHBOURHOOD_158_Etobicoke City Centre',
 'NEIGHBOURHOOD_158_Etobicoke West Mall',
 'NEIGHBOURHOOD_158_Fenside-Parkwoods',
 'NEIGHBOURHOOD_158_Flemingdon Park',
 'NEIGHBOURHOOD_158_Forest Hill North',
 'NEIGHBOURHOOD_158_Forest Hill South',
 'NEIGHBOURHOOD_158_Fort York-Liberty Village',
 'NEIGHBOURHOOD_158_Glenfield-Jane Heights',
 'NEIGHBOURHOOD_158_Golfdale-Cedarbrae-Woburn',
 'NEIGHBOURHOOD_158_Greenwood-Coxwell',
 'NEIGHBOURHOOD_158_Guildwood',
 'NEIGHBOURHOOD_158_Harbourfront-CityPlace',
 'NEIGHBOURHOOD_158_Henry Farm',
 'NEIGHBOURHOOD_158_High Park North',
 'NEIGHBOURHOOD_158_High Park-Swansea',
 'NEIGHBOURHOOD_158_Highland Creek',
 'NEIGHBOURHOOD_158_Hillcrest Village',
 'NEIGHBOURHOOD_158_Humber Bay Shores',
 'NEIGHBOURHOOD_158_Humber Heights-Westmount',
 'NEIGHBOURHOOD_158_Humber Summit',
 'NEIGHBOURHOOD_158_Humbermede',
 'NEIGHBOURHOOD_158_Humewood-Cedarvale',
 'NEIGHBOURHOOD_158_Ionview',
 'NEIGHBOURHOOD_158_Islington',
 'NEIGHBOURHOOD_158_Junction Area',
 'NEIGHBOURHOOD_158_Junction-Wallace Emerson',
 'NEIGHBOURHOOD_158_Keelesdale-Eglinton West',
 'NEIGHBOURHOOD_158_Kennedy Park',
 'NEIGHBOURHOOD_158_Kensington-Chinatown',
 'NEIGHBOURHOOD_158_Kingsview Village-The Westway',
 'NEIGHBOURHOOD_158_Kingsway South',
 "NEIGHBOURHOOD_158_L'Amoreaux West",
 'NEIGHBOURHOOD_158_Lambton Baby Point',
 'NEIGHBOURHOOD_158_Lansing-Westgate',
 'NEIGHBOURHOOD_158_Lawrence Park North',
 'NEIGHBOURHOOD_158_Lawrence Park South',
 'NEIGHBOURHOOD_158_Leaside-Bennington',
 'NEIGHBOURHOOD_158_Little Portugal',
 'NEIGHBOURHOOD_158_Long Branch',
 'NEIGHBOURHOOD_158_Malvern East',
 'NEIGHBOURHOOD_158_Malvern West',
 'NEIGHBOURHOOD_158_Maple Leaf',
 'NEIGHBOURHOOD_158_Markland Wood',
 'NEIGHBOURHOOD_158_Milliken',
 'NEIGHBOURHOOD_158_Mimico-Queensway',
 'NEIGHBOURHOOD_158_Morningside',
 'NEIGHBOURHOOD_158_Morningside Heights',
 'NEIGHBOURHOOD_158_Moss Park',
 'NEIGHBOURHOOD_158_Mount Dennis',
 'NEIGHBOURHOOD_158_Mount Olive-Silverstone-Jamestown',
 'NEIGHBOURHOOD_158_Mount Pleasant East',
 'NEIGHBOURHOOD_158_New Toronto',
 'NEIGHBOURHOOD_158_Newtonbrook East',
 'NEIGHBOURHOOD_158_Newtonbrook West',
 'NEIGHBOURHOOD_158_North Riverdale',
 'NEIGHBOURHOOD_158_North St.James Town',
 'NEIGHBOURHOOD_158_North Toronto',
 "NEIGHBOURHOOD_158_O'Connor-Parkview",
 'NEIGHBOURHOOD_158_Oakdale-Beverley Heights',
 'NEIGHBOURHOOD_158_Oakridge',
 'NEIGHBOURHOOD_158_Oakwood Village',
 'NEIGHBOURHOOD_158_Old East York',
 'NEIGHBOURHOOD_158_Palmerston-Little Italy',
 "NEIGHBOURHOOD_158_Parkwoods-O'Connor Hills",
 'NEIGHBOURHOOD_158_Pelmo Park-Humberlea',
 'NEIGHBOURHOOD_158_Playter Estates-Danforth',
 'NEIGHBOURHOOD_158_Pleasant View',
 'NEIGHBOURHOOD_158_Princess-Rosethorn',
 'NEIGHBOURHOOD_158_Regent Park',
 'NEIGHBOURHOOD_158_Rexdale-Kipling',
 'NEIGHBOURHOOD_158_Rockcliffe-Smythe',
 'NEIGHBOURHOOD_158_Roncesvalles',
 'NEIGHBOURHOOD_158_Rosedale-Moore Park',
 'NEIGHBOURHOOD_158_Runnymede-Bloor West Village',
 'NEIGHBOURHOOD_158_Rustic',
 'NEIGHBOURHOOD_158_Scarborough Village',
 'NEIGHBOURHOOD_158_South Eglinton-Davisville',
 'NEIGHBOURHOOD_158_South Parkdale',
 'NEIGHBOURHOOD_158_South Riverdale',
 'NEIGHBOURHOOD_158_St Lawrence-East Bayfront-The Islands ',
 'NEIGHBOURHOOD_158_St.Andrew-Windfields',
 'NEIGHBOURHOOD_158_Steeles',
 'NEIGHBOURHOOD_158_Stonegate-Queensway',
 "NEIGHBOURHOOD_158_Tam O'Shanter-Sullivan",
 'NEIGHBOURHOOD_158_Taylor-Massey',
 'NEIGHBOURHOOD_158_The Beaches',
 'NEIGHBOURHOOD_158_Thistletown-Beaumond Heights',
 'NEIGHBOURHOOD_158_Thorncliffe Park',
 'NEIGHBOURHOOD_158_Trinity-Bellwoods',
 'NEIGHBOURHOOD_158_University',
 'NEIGHBOURHOOD_158_Victoria Village',
 'NEIGHBOURHOOD_158_Wellington Place',
 'NEIGHBOURHOOD_158_West Hill',
 'NEIGHBOURHOOD_158_West Humber-Clairville',
 'NEIGHBOURHOOD_158_West Queen West',
 'NEIGHBOURHOOD_158_West Rouge',
 'NEIGHBOURHOOD_158_Westminster-Branson',
 'NEIGHBOURHOOD_158_Weston',
 'NEIGHBOURHOOD_158_Weston-Pelham Park',
 'NEIGHBOURHOOD_158_Wexford/Maryvale',
 'NEIGHBOURHOOD_158_Willowdale West',
 'NEIGHBOURHOOD_158_Willowridge-Martingrove-Richview',
 'NEIGHBOURHOOD_158_Woburn North',
 'NEIGHBOURHOOD_158_Woodbine Corridor',
 'NEIGHBOURHOOD_158_Woodbine-Lumsden',
 'NEIGHBOURHOOD_158_Wychwood',
 'NEIGHBOURHOOD_158_Yonge-Bay Corridor',
 'NEIGHBOURHOOD_158_Yonge-Doris',
 'NEIGHBOURHOOD_158_Yonge-Eglinton',
 'NEIGHBOURHOOD_158_Yonge-St.Clair',
 'NEIGHBOURHOOD_158_York University Heights',
 'NEIGHBOURHOOD_158_Yorkdale-Glen Park']

def create_neighborhood_one_hot_df(neighborhood_name, neighborhood_list):
    # Create a list of columns with 'NEIGHBOURHOOD_158_' prefix and neighborhood name
    columns = [col for col in neighborhood_list]
    
    # Initialize a dictionary with all values set to 0
    data = {col: 0 for col in columns}
    
    # Set the input neighborhood to 1
    if f'NEIGHBOURHOOD_158_{neighborhood_name}' in data:
        data[f'NEIGHBOURHOOD_158_{neighborhood_name}'] = 1
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    
    return df


time_range_list = ['OCC_TIME_RANGE_Afternoon',
'OCC_TIME_RANGE_Evening',
'OCC_TIME_RANGE_Morning',
'OCC_TIME_RANGE_Night']

def create_time_range_one_hot_df(time_range, time_range_list):
    # Initialize a dictionary with all time ranges set to 0
    data = {col: 0 for col in time_range_list}
    # print(f'OCC_TIME_RANGE_{time_range}')
   

    
    # Set the input time range to 1
    if f'OCC_TIME_RANGE_{time_range}' in data:
        
        data[f'OCC_TIME_RANGE_{time_range}'] = 1
        
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    
    return df

weather_list = [
    'Weather_Clear',
    'Weather_Cloudy',
    'Weather_Foggy',
    'Weather_Rainy',
    'Weather_Rainy/Foggy',
    'Weather_Snowy',
    'Weather_Snowy/Foggy',
    'Weather_Stormy'
]
def create_weather_one_hot_df(weather, weather_list):
    # Initialize a dictionary with all weather conditions set to 0
    data = {col: 0 for col in weather_list}
    

    # Set the input weather condition to 1
    if f'Weather_{weather}' in data:
        data[f'Weather_{weather}'] = 1
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    
    return df

def get_income_category_features(income):
    # Create a row of zeroes for the three income category columns
    income_features = {'income_category_Medium': 0, 'income_category_High': 0, 'income_category_Very High': 0}
    
    # Determine the income category and set the corresponding feature to 1
    if 30000 < income <= 60000:
        income_features['income_category_Medium'] = 1
    elif 60000 < income <= 100000:
        income_features['income_category_High'] = 1
    elif income > 100000:
        income_features['income_category_Very High'] = 1
    
    return income_features


def createNewRecord(neighborhood_name, city_name, api_key, weather_list, neighborhoodList, time_range_list):
    # weather related values
    city_name=city_name
    weather_data = get_weather_by_city(api_key, city_name)
    weather_data["weather_description"] = map_weather_description(weather_data["weather_description"])
    Temp_C, Rel_Hum, Wind_Spd, Visibility, weather = weather_data['temperature'], weather_data['humidity'],weather_data['wind_speed'], weather_data['visibility'], weather_data['weather_description']
    one_hot_weather = create_weather_one_hot_df(weather, weather_list)
    Temp_Wind_Interaction = Temp_C * Wind_Spd

    # Neighborhood related values
    NEIGH_POPULATION = neighbor_df[neighborhood_name].NEIGH_POPULATION
    NEIGH_AVG_AGE = neighbor_df[neighborhood_name].NEIGH_AVG_AGE
    NEIGH_UNEMP_RATE = neighbor_df[neighborhood_name].NEIGH_UNEMP_RATE
    NEIGH_AVG_HSHLD_SIZE = neighbor_df[neighborhood_name].NEIGH_AVG_HSHLD_SIZE
    NEIGH_AVG_INCM = neighbor_df[neighborhood_name].NEIGH_AVG_INCM
    NEIGH_POP_INCOME_INTERACTION = NEIGH_POPULATION * NEIGH_AVG_INCM
    one_hot_neighbor = create_neighborhood_one_hot_df(neighborhood_name, neighborhoodList)
    one_hot_income = get_income_category_features(NEIGH_AVG_INCM)
    one_hot_income=pd.DataFrame([one_hot_income])

    # Date related values
    is_weekend, DOW_sin, DOW_cos = getWeek()
    now = datetime.now()
    current_hour = now.hour
    OCC_TIME_RANGE = categorize_time(current_hour)
    one_hot_timeRange = create_time_range_one_hot_df(OCC_TIME_RANGE, time_range_list)

    # Creating the final dataframe
    final_df = pd.DataFrame({
        'Temp_C': [Temp_C],
        'Rel_Hum': [Rel_Hum],
        'Wind_Spd': [Wind_Spd],
        'Visibility': [Visibility],
        'Temp_Wind_Interaction': [Temp_Wind_Interaction],
        'NEIGH_POPULATION': [NEIGH_POPULATION],
        'NEIGH_AVG_AGE': [NEIGH_AVG_AGE],
        'NEIGH_UNEMP_RATE': [NEIGH_UNEMP_RATE],
        'NEIGH_AVG_HSHLD_SIZE': [NEIGH_AVG_HSHLD_SIZE],
        'NEIGH_AVG_INCM': [NEIGH_AVG_INCM],
        'NEIGH_POP_INCOME_INTERACTION': [NEIGH_POP_INCOME_INTERACTION],
        'is_weekend': [is_weekend],
        'DOW_sin': [DOW_sin],
        'DOW_cos': [DOW_cos]
    })
    
    # Adding one-hot encoded columns
    final_df = pd.concat([final_df, one_hot_weather, one_hot_neighbor, one_hot_income, one_hot_timeRange], axis=1)

    return final_df


# testInput=createNewRecord(neighborhood_name="Rexdale-Kipling", city_name="Toronto", api_key=api_key, weather_list=weather_list, neighborhoodList=neighborhoodList, time_range_list=time_range_list)

# print(testInput)

columnOrder=['Temp_C',
 'Rel_Hum',
 'Wind_Spd',
 'Visibility',
 'NEIGH_POPULATION',
 'NEIGH_AVG_AGE',
 'NEIGH_UNEMP_RATE',
 'NEIGH_AVG_HSHLD_SIZE',
 'NEIGH_AVG_INCM',
 'NEIGH_POP_INCOME_INTERACTION',
 'is_weekend',
 'Temp_Wind_Interaction',
 'income_category_Medium',
 'income_category_High',
 'income_category_Very High',
 'DOW_sin',
 'DOW_cos',
 'NEIGHBOURHOOD_158_Agincourt North',
 'NEIGHBOURHOOD_158_Agincourt South-Malvern West',
 'NEIGHBOURHOOD_158_Alderwood',
 'NEIGHBOURHOOD_158_Annex',
 'NEIGHBOURHOOD_158_Avondale',
 'NEIGHBOURHOOD_158_Banbury-Don Mills',
 'NEIGHBOURHOOD_158_Bathurst Manor',
 'NEIGHBOURHOOD_158_Bay-Cloverhill',
 'NEIGHBOURHOOD_158_Bayview Village',
 'NEIGHBOURHOOD_158_Bayview Woods-Steeles',
 'NEIGHBOURHOOD_158_Bedford Park-Nortown',
 'NEIGHBOURHOOD_158_Beechborough-Greenbrook',
 'NEIGHBOURHOOD_158_Bendale South',
 'NEIGHBOURHOOD_158_Bendale-Glen Andrew',
 'NEIGHBOURHOOD_158_Birchcliffe-Cliffside',
 'NEIGHBOURHOOD_158_Black Creek',
 'NEIGHBOURHOOD_158_Blake-Jones',
 'NEIGHBOURHOOD_158_Briar Hill-Belgravia',
 'NEIGHBOURHOOD_158_Bridle Path-Sunnybrook-York Mills',
 'NEIGHBOURHOOD_158_Broadview North',
 'NEIGHBOURHOOD_158_Brookhaven-Amesbury',
 'NEIGHBOURHOOD_158_Cabbagetown-South St.James Town',
 'NEIGHBOURHOOD_158_Caledonia-Fairbank',
 'NEIGHBOURHOOD_158_Casa Loma',
 'NEIGHBOURHOOD_158_Centennial Scarborough',
 'NEIGHBOURHOOD_158_Church-Wellesley',
 'NEIGHBOURHOOD_158_Clairlea-Birchmount',
 'NEIGHBOURHOOD_158_Clanton Park',
 'NEIGHBOURHOOD_158_Cliffcrest',
 'NEIGHBOURHOOD_158_Corso Italia-Davenport',
 'NEIGHBOURHOOD_158_Danforth',
 'NEIGHBOURHOOD_158_Danforth East York',
 'NEIGHBOURHOOD_158_Don Valley Village',
 'NEIGHBOURHOOD_158_Dorset Park',
 'NEIGHBOURHOOD_158_Dovercourt Village',
 'NEIGHBOURHOOD_158_Downsview',
 'NEIGHBOURHOOD_158_Downtown Yonge East',
 'NEIGHBOURHOOD_158_Dufferin Grove',
 'NEIGHBOURHOOD_158_East End-Danforth',
 "NEIGHBOURHOOD_158_East L'Amoreaux",
 'NEIGHBOURHOOD_158_East Willowdale',
 'NEIGHBOURHOOD_158_Edenbridge-Humber Valley',
 'NEIGHBOURHOOD_158_Eglinton East',
 'NEIGHBOURHOOD_158_Elms-Old Rexdale',
 'NEIGHBOURHOOD_158_Englemount-Lawrence',
 'NEIGHBOURHOOD_158_Eringate-Centennial-West Deane',
 'NEIGHBOURHOOD_158_Etobicoke City Centre',
 'NEIGHBOURHOOD_158_Etobicoke West Mall',
 'NEIGHBOURHOOD_158_Fenside-Parkwoods',
 'NEIGHBOURHOOD_158_Flemingdon Park',
 'NEIGHBOURHOOD_158_Forest Hill North',
 'NEIGHBOURHOOD_158_Forest Hill South',
 'NEIGHBOURHOOD_158_Fort York-Liberty Village',
 'NEIGHBOURHOOD_158_Glenfield-Jane Heights',
 'NEIGHBOURHOOD_158_Golfdale-Cedarbrae-Woburn',
 'NEIGHBOURHOOD_158_Greenwood-Coxwell',
 'NEIGHBOURHOOD_158_Guildwood',
 'NEIGHBOURHOOD_158_Harbourfront-CityPlace',
 'NEIGHBOURHOOD_158_Henry Farm',
 'NEIGHBOURHOOD_158_High Park North',
 'NEIGHBOURHOOD_158_High Park-Swansea',
 'NEIGHBOURHOOD_158_Highland Creek',
 'NEIGHBOURHOOD_158_Hillcrest Village',
 'NEIGHBOURHOOD_158_Humber Bay Shores',
 'NEIGHBOURHOOD_158_Humber Heights-Westmount',
 'NEIGHBOURHOOD_158_Humber Summit',
 'NEIGHBOURHOOD_158_Humbermede',
 'NEIGHBOURHOOD_158_Humewood-Cedarvale',
 'NEIGHBOURHOOD_158_Ionview',
 'NEIGHBOURHOOD_158_Islington',
 'NEIGHBOURHOOD_158_Junction Area',
 'NEIGHBOURHOOD_158_Junction-Wallace Emerson',
 'NEIGHBOURHOOD_158_Keelesdale-Eglinton West',
 'NEIGHBOURHOOD_158_Kennedy Park',
 'NEIGHBOURHOOD_158_Kensington-Chinatown',
 'NEIGHBOURHOOD_158_Kingsview Village-The Westway',
 'NEIGHBOURHOOD_158_Kingsway South',
 "NEIGHBOURHOOD_158_L'Amoreaux West",
 'NEIGHBOURHOOD_158_Lambton Baby Point',
 'NEIGHBOURHOOD_158_Lansing-Westgate',
 'NEIGHBOURHOOD_158_Lawrence Park North',
 'NEIGHBOURHOOD_158_Lawrence Park South',
 'NEIGHBOURHOOD_158_Leaside-Bennington',
 'NEIGHBOURHOOD_158_Little Portugal',
 'NEIGHBOURHOOD_158_Long Branch',
 'NEIGHBOURHOOD_158_Malvern East',
 'NEIGHBOURHOOD_158_Malvern West',
 'NEIGHBOURHOOD_158_Maple Leaf',
 'NEIGHBOURHOOD_158_Markland Wood',
 'NEIGHBOURHOOD_158_Milliken',
 'NEIGHBOURHOOD_158_Mimico-Queensway',
 'NEIGHBOURHOOD_158_Morningside',
 'NEIGHBOURHOOD_158_Morningside Heights',
 'NEIGHBOURHOOD_158_Moss Park',
 'NEIGHBOURHOOD_158_Mount Dennis',
 'NEIGHBOURHOOD_158_Mount Olive-Silverstone-Jamestown',
 'NEIGHBOURHOOD_158_Mount Pleasant East',
 'NEIGHBOURHOOD_158_New Toronto',
 'NEIGHBOURHOOD_158_Newtonbrook East',
 'NEIGHBOURHOOD_158_Newtonbrook West',
 'NEIGHBOURHOOD_158_North Riverdale',
 'NEIGHBOURHOOD_158_North St.James Town',
 'NEIGHBOURHOOD_158_North Toronto',
 "NEIGHBOURHOOD_158_O'Connor-Parkview",
 'NEIGHBOURHOOD_158_Oakdale-Beverley Heights',
 'NEIGHBOURHOOD_158_Oakridge',
 'NEIGHBOURHOOD_158_Oakwood Village',
 'NEIGHBOURHOOD_158_Old East York',
 'NEIGHBOURHOOD_158_Palmerston-Little Italy',
 "NEIGHBOURHOOD_158_Parkwoods-O'Connor Hills",
 'NEIGHBOURHOOD_158_Pelmo Park-Humberlea',
 'NEIGHBOURHOOD_158_Playter Estates-Danforth',
 'NEIGHBOURHOOD_158_Pleasant View',
 'NEIGHBOURHOOD_158_Princess-Rosethorn',
 'NEIGHBOURHOOD_158_Regent Park',
 'NEIGHBOURHOOD_158_Rexdale-Kipling',
 'NEIGHBOURHOOD_158_Rockcliffe-Smythe',
 'NEIGHBOURHOOD_158_Roncesvalles',
 'NEIGHBOURHOOD_158_Rosedale-Moore Park',
 'NEIGHBOURHOOD_158_Runnymede-Bloor West Village',
 'NEIGHBOURHOOD_158_Rustic',
 'NEIGHBOURHOOD_158_Scarborough Village',
 'NEIGHBOURHOOD_158_South Eglinton-Davisville',
 'NEIGHBOURHOOD_158_South Parkdale',
 'NEIGHBOURHOOD_158_South Riverdale',
 'NEIGHBOURHOOD_158_St Lawrence-East Bayfront-The Islands ',
 'NEIGHBOURHOOD_158_St.Andrew-Windfields',
 'NEIGHBOURHOOD_158_Steeles',
 'NEIGHBOURHOOD_158_Stonegate-Queensway',
 "NEIGHBOURHOOD_158_Tam O'Shanter-Sullivan",
 'NEIGHBOURHOOD_158_Taylor-Massey',
 'NEIGHBOURHOOD_158_The Beaches',
 'NEIGHBOURHOOD_158_Thistletown-Beaumond Heights',
 'NEIGHBOURHOOD_158_Thorncliffe Park',
 'NEIGHBOURHOOD_158_Trinity-Bellwoods',
 'NEIGHBOURHOOD_158_University',
 'NEIGHBOURHOOD_158_Victoria Village',
 'NEIGHBOURHOOD_158_Wellington Place',
 'NEIGHBOURHOOD_158_West Hill',
 'NEIGHBOURHOOD_158_West Humber-Clairville',
 'NEIGHBOURHOOD_158_West Queen West',
 'NEIGHBOURHOOD_158_West Rouge',
 'NEIGHBOURHOOD_158_Westminster-Branson',
 'NEIGHBOURHOOD_158_Weston',
 'NEIGHBOURHOOD_158_Weston-Pelham Park',
 'NEIGHBOURHOOD_158_Wexford/Maryvale',
 'NEIGHBOURHOOD_158_Willowdale West',
 'NEIGHBOURHOOD_158_Willowridge-Martingrove-Richview',
 'NEIGHBOURHOOD_158_Woburn North',
 'NEIGHBOURHOOD_158_Woodbine Corridor',
 'NEIGHBOURHOOD_158_Woodbine-Lumsden',
 'NEIGHBOURHOOD_158_Wychwood',
 'NEIGHBOURHOOD_158_Yonge-Bay Corridor',
 'NEIGHBOURHOOD_158_Yonge-Doris',
 'NEIGHBOURHOOD_158_Yonge-Eglinton',
 'NEIGHBOURHOOD_158_Yonge-St.Clair',
 'NEIGHBOURHOOD_158_York University Heights',
 'NEIGHBOURHOOD_158_Yorkdale-Glen Park',
 'Weather_Clear',
 'Weather_Cloudy',
 'Weather_Foggy',
 'Weather_Rainy',
 'Weather_Rainy/Foggy',
 'Weather_Snowy',
 'Weather_Snowy/Foggy',
 'Weather_Stormy',
 'OCC_TIME_RANGE_Afternoon',
 'OCC_TIME_RANGE_Evening',
 'OCC_TIME_RANGE_Morning',
 'OCC_TIME_RANGE_Night']

# testInput = testInput[columnOrder]

df_train=pd.read_parquet('columnNeighborhood.parquet')

from sklearn.preprocessing import StandardScaler

# Assume you have training data df_train (or any DataFrame you used for model training)

# Define the features to scale
features_to_scale = ['Temp_C', 'Rel_Hum', 'Wind_Spd', 'Visibility',
                     'NEIGH_POPULATION', 'NEIGH_AVG_AGE', 'NEIGH_UNEMP_RATE',
                     'NEIGH_AVG_HSHLD_SIZE', 'NEIGH_AVG_INCM', 
                     'NEIGH_POP_INCOME_INTERACTION', 'Temp_Wind_Interaction',
                     'DOW_sin', 'DOW_cos']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data
# Assuming 'df_train' is your training DataFrame
scaler.fit(df_train[features_to_scale])  # Fit on the training set

# testInput_scaled = testInput.copy()  # Copy to avoid modifying the original DataFrame
# testInput_scaled[features_to_scale] = scaler.transform(testInput_scaled[features_to_scale])

from tensorflow.keras.models import load_model


best_model = load_model('best_model.keras')

# Example input data (replace with your own test data)
# input_data = testInput_scaled  # This should be the preprocessed input data for prediction

# Step 3: Make predictions using the model
# predictions = best_model.predict(input_data)

# If you have a classification problem, the predictions might be probabilities, so you may want to apply an argmax
# to get the class with the highest probability
# predicted_classes = np.argmax(predictions, axis=1)

# If it's a regression problem, the predictions will directly be numerical values
# predicted_values = predictions

# Step 4: Output the predictions
# print(predicted_classes)  # For classification problems

df_neighborhoods = neighbor_df.columns.tolist()

from fuzzywuzzy import process

# Function to normalize neighborhood names
def normalize_name(name):
    """
    Normalize a name by:
    - Converting to lowercase.
    - Replacing spaces and underscores with hyphens.
    - Removing special characters.
    """
    import re
    name = name.lower()
    name = re.sub(r'[ _]', '-', name)
    name = re.sub(r'[^\w\-]', '', name)
    return name
# Preprocess neighborhoods into a dictionary
preprocessed_neighborhoods = {
    normalize_name(name): name for name in df_neighborhoods
}


def fuzzy_match_neighborhood(api_name, preprocessed_neighborhoods, normalize_func, threshold=80):
    
    normalized_name = normalize_func(api_name)
    # Extract the best match using the preprocessed dictionary keys
    match, score = process.extractOne(normalized_name, list(preprocessed_neighborhoods.keys()))
    if score >= threshold:
        return preprocessed_neighborhoods[match]  # Retrieve the original name
    return None



# api_neighborhoods={'Leaside', 'Leaside-Bennington', 'Don Mills', 'Rosedale-Moore Park', 'Downtown Yonge', 'Henry Farm', 'Garden District', 'Rosedale', "Governor's Bridge", 'Banbury-Don Mills', 'Parkway Forest'}

# def get_route_predictions(
#     routes, 
#     df_neighborhoods, 
#     model, 
#     normalize_func, 
#     scaler, 
#     features_to_scale, 
#     columnOrder, 
#     threshold=80, 
#     confidence_threshold=0.6
# ):
#     route_predictions = []
#     batch_inputs = []
#     batch_indices = []
#     batch_neighborhoods = []

#     # Step 1: Preprocess all neighborhoods and collect inputs for batch prediction
#     for route_idx, route in enumerate(routes):
#         for api_name in route:
#             matched_name = fuzzy_match_neighborhood(api_name, df_neighborhoods, normalize_func, threshold)
#             if matched_name:
#                 input = createNewRecord(
#                     neighborhood_name=matched_name,
#                     city_name="Toronto",
#                     api_key=api_key,
#                     weather_list=weather_list,
#                     neighborhoodList=neighborhoodList,
#                     time_range_list=time_range_list
#                 )
                
#                 input = input[columnOrder]
#                 input_flat = input.iloc[0].to_dict()  # Flatten the DataFrame
#                 batch_inputs.append(input_flat) 
                
#                 batch_indices.append((route_idx, len(batch_neighborhoods)))  # Map to route/neighborhood
#                 batch_neighborhoods.append(matched_name)
#             else:
#                 print(f"No match found for neighborhood: {api_name} in route: {route}")

#     # Step 2: Scale features in batch
#     if batch_inputs:
#         batch_inputs_df = pd.DataFrame(batch_inputs, columns=columnOrder)
#         batch_inputs_df[features_to_scale] = scaler.transform(batch_inputs_df[features_to_scale])

#         # Step 3: Perform batch predictions
#         batch_predictions = model.predict(batch_inputs_df)
#         batch_confidence_values = np.max(batch_predictions, axis=1)
#         batch_predicted_classes = np.argmax(batch_predictions, axis=1)

#     # Step 4: Reconstruct predictions for each route
#     route_data = {i: {"predicted_classes": [], "confidence_values": []} for i in range(len(routes))}

#     for idx, (route_idx, neighborhood_idx) in enumerate(batch_indices):
#         predicted_class = batch_predicted_classes[idx]
#         confidence_value = batch_confidence_values[idx]

#         # Upgrade low-confidence medium risk (class 0) predictions to high risk (class 1)
#         if predicted_class == 0 and confidence_value < confidence_threshold:
#             predicted_class = 1

#         route_data[route_idx]["predicted_classes"].append(predicted_class)
#         route_data[route_idx]["confidence_values"].append(confidence_value)

#     # Step 5: Calculate safety score and overall route class
#     for route_idx, route in enumerate(routes):
#         predicted_classes = route_data[route_idx]["predicted_classes"]
#         confidence_values = route_data[route_idx]["confidence_values"]

#         route_prediction = {
#             'route': route,
#             'predicted_classes': predicted_classes,
#             'confidence_values': confidence_values,
#         }

#         if predicted_classes:
#             # Calculate granular safety score
#             weighted_risks = [
#                 (class_ + 1) * confidence 
#                 for class_, confidence in zip(predicted_classes, confidence_values)
#             ]
#             route_prediction['safety_score'] = round(sum(weighted_risks) / len(predicted_classes), 3)
            
#             # Determine the overall predicted class
#             route_prediction['overall_predicted_class'] = np.argmax(np.bincount(predicted_classes))
            
#             # Calculate overall confidence for the overall predicted class
#             route_prediction['overall_confidence'] = np.mean([
#                 confidence 
#                 for class_, confidence in zip(predicted_classes, confidence_values)
#                 if class_ == route_prediction['overall_predicted_class']
#             ])
#         else:
#             route_prediction['safety_score'] = None
#             route_prediction['overall_predicted_class'] = None
#             route_prediction['overall_confidence'] = None

#         route_predictions.append(route_prediction)

#     # Step 6: Sort routes by safety score
#     route_predictions.sort(key=lambda x: (x['safety_score'] if x['safety_score'] is not None else float('inf')))
#     return route_predictions

def get_route_predictions(
    routes, 
    df_neighborhoods, 
    model, 
    normalize_func, 
    scaler, 
    features_to_scale, 
    columnOrder, 
    threshold=80, 
    confidence_threshold=0.6
):
    route_predictions = []
    batch_inputs = []
    batch_indices = []
    batch_neighborhoods = []
    batch_create_inputs = []  # For batch processing `createNewRecord`

    # Step 1: Preprocess all neighborhoods and collect inputs for batch prediction
    for route_idx, route in enumerate(routes):
        for api_name in route:
            matched_name = fuzzy_match_neighborhood(api_name, df_neighborhoods, normalize_func, threshold)
            if matched_name:
                batch_create_inputs.append({
                    "neighborhood_name": matched_name,
                    "city_name": "Toronto",
                    "api_key": api_key,
                    "weather_list": weather_list,
                    "neighborhoodList": neighborhoodList,
                    "time_range_list": time_range_list
                })
                batch_indices.append((route_idx, len(batch_neighborhoods)))  # Map to route/neighborhood
                batch_neighborhoods.append(matched_name)
            # else:
                # print(f"No match found for neighborhood: {api_name} in route: {route}")

    # Step 2: Create new records in batch
    if batch_create_inputs:
        batch_created_records = createNewRecordBatch(batch_create_inputs)  # Batch function for creating records
        batch_created_records_df = pd.DataFrame(batch_created_records, columns=columnOrder)

        # Step 3: Scale features in batch
        batch_created_records_df[features_to_scale] = scaler.transform(batch_created_records_df[features_to_scale])

        # Step 4: Perform batch predictions
        batch_predictions = model.predict(batch_created_records_df)
        batch_confidence_values = np.max(batch_predictions, axis=1)
        batch_predicted_classes = np.argmax(batch_predictions, axis=1)

        # Step 5: Reconstruct predictions for each route
        route_data = {i: {"predicted_classes": [], "confidence_values": []} for i in range(len(routes))}

        for idx, (route_idx, neighborhood_idx) in enumerate(batch_indices):
            predicted_class = batch_predicted_classes[idx]
            confidence_value = batch_confidence_values[idx]

            # Upgrade low-confidence medium risk (class 0) predictions to high risk (class 1)
            if predicted_class == 0 and confidence_value < confidence_threshold:
                predicted_class = 1

            route_data[route_idx]["predicted_classes"].append(predicted_class)
            route_data[route_idx]["confidence_values"].append(confidence_value)

        # Step 6: Calculate safety score and overall route class
        for route_idx, route in enumerate(routes):
            predicted_classes = route_data[route_idx]["predicted_classes"]
            confidence_values = route_data[route_idx]["confidence_values"]

            route_prediction = {
                'route': route,
                'predicted_classes': predicted_classes,
                'confidence_values': confidence_values,
            }

            if predicted_classes:
                # Calculate granular safety score
                weighted_risks = [
                    (class_ + 1) * confidence 
                    for class_, confidence in zip(predicted_classes, confidence_values)
                ]
                route_prediction['safety_score'] = round(sum(weighted_risks) / len(predicted_classes), 3)
                
                # Determine the overall predicted class
                route_prediction['overall_predicted_class'] = np.argmax(np.bincount(predicted_classes))
                
                # Calculate overall confidence for the overall predicted class
                route_prediction['overall_confidence'] = np.mean([
                    confidence 
                    for class_, confidence in zip(predicted_classes, confidence_values)
                    if class_ == route_prediction['overall_predicted_class']
                ])
            else:
                route_prediction['safety_score'] = None
                route_prediction['overall_predicted_class'] = None
                route_prediction['overall_confidence'] = None

            route_predictions.append(route_prediction)

    # Step 7: Sort routes by safety score
    route_predictions.sort(key=lambda x: (x['safety_score'] if x['safety_score'] is not None else float('inf')))
    return route_predictions

def createNewRecordBatch(inputs):
    """
    Simulate batch processing for createNewRecord.
    Each input in `inputs` should be a dictionary containing the parameters for `createNewRecord`.
    """
    records = []
    for input_params in inputs:
        record = createNewRecord(
            neighborhood_name=input_params["neighborhood_name"],
            city_name=input_params["city_name"],
            api_key=input_params["api_key"],
            weather_list=input_params["weather_list"],
            neighborhoodList=input_params["neighborhoodList"],
            time_range_list=input_params["time_range_list"]
        )
        records.append(record.iloc[0].to_dict())  # Flatten each record DataFrame to a dictionary
    return records

@app.route('/predict_routes', methods=['POST'])
def predict_routes():
    print("Data get?")
    data = request.get_json()
    neighborhoods_by_route = data.get('neighborhoods_by_route', [])
    print(neighborhoods_by_route)

    if not neighborhoods_by_route:
        return jsonify({'error': 'No neighborhoods data received'}), 400

   
    predictions = get_route_predictions(
            routes=neighborhoods_by_route,
            df_neighborhoods=preprocessed_neighborhoods,
            model=best_model,
            normalize_func=normalize_name,
            scaler=scaler,
            features_to_scale=features_to_scale,
            columnOrder=columnOrder
        )
    # print(predictions)

    extracted_predictions = {}
    for index, prediction in enumerate(predictions):
        route_label = f"route_{index + 1}"  # Label routes as route_1, route_2, ...
        extracted_predictions[route_label] = {
            'overall_predicted_class': int(prediction['overall_predicted_class']) if prediction['overall_predicted_class'] is not None else -1,  # Default to -1 for no prediction
            'overall_confidence': float(prediction['overall_confidence']) if prediction['overall_confidence'] is not None else 0.0,  # Default to 0.0 for no confidence
            'risk_score': float(prediction['safety_score']) if prediction['safety_score'] is not None else float('inf')  # Default to 'inf' for no safety score
        }


    # print(extracted_predictions)

    
   

    return jsonify({'predictions': extracted_predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
