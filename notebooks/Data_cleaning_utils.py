import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def column_name_change(data : pd.DataFrame):

        return (
            data.rename(str.lower,axis=1)
           .rename({'delivery_person_id':'rider_id', 'delivery_person_age':'age',
           'delivery_person_ratings':'ratings', 'restaurant_latitude':'restaurant_lat',
           'restaurant_longitude':'restaurant_long', 'delivery_location_latitude':'delivery_lat',
           'delivery_location_longitude':'delivery_long','time_orderd':'order_time','time_order_picked':'picked_time', 
           'weatherconditions':'weather', 'road_traffic_density':'traffic','type_of_order':'order_type',
           'type_of_vehicle':'vehicle_type','city':'city_type', 'time_taken(min)':'time'},axis=1)
              )


def data_cleaning(data : pd.DataFrame):

        minor_rider = data.loc[data['age'].astype("float") < 18]
        six_star_ratings = data.loc[data['ratings'].astype("float") > 5.0]


        return(
        
               data.replace(['NaN ','conditions NaN'], np.nan)
               .drop(pd.concat([minor_rider,six_star_ratings]).index,axis=0)
               .assign(
            
                city_name = lambda x : x['rider_id'].str.split("RES").str.get(0).str.strip(),
                age = lambda x : x['age'].astype("float"),
                age_cat = lambda x : pd.cut(x['age'].astype("float"), bins = [18, 25, 35, 50], labels = ['young', 'young-adult', 'Middle-aged']),
                ratings = lambda x : x['ratings'].astype("float"),
                ratings_cat = lambda x : pd.cut(x['ratings'].astype("float"),bins=[1,4,4.5,5],labels=['less than 4','4-4.5','4.5-5'],include_lowest=True),
                weather = lambda x : x['weather'].str.strip().str.split('conditions ').str.get(1).str.lower(),
                traffic = lambda x : x['traffic'].str.strip().str.lower(),
                vehicle_condition = lambda x : x['vehicle_condition'].map({0:3, 1:2,2:1, 3:0}),
                order_type = lambda x : x['order_type'].str.strip().str.lower(),
                vehicle_type = lambda x : x['vehicle_type'].str.strip().str.lower(),
                multiple_deliveries = lambda x : x['multiple_deliveries'].astype("float"),
                festival = lambda x : x['festival'].str.strip().str.lower(),
                city_type = lambda x : x['city_type'].str.strip().str.lower(),
                time = lambda x : x['time'].str.strip().str.split(" ").str.get(1).astype("float")
            
            )
        
        )


def clean_lat_long(data: pd.DataFrame, threshold =1):

        locations_subset = ['restaurant_lat', 'restaurant_long',
                                      'delivery_lat', 'delivery_long']
        locations_abs = data.loc[:,locations_subset].abs()

        return (
                data.assign(**{
                    cols: np.where(locations_abs[cols] < threshold, np.nan ,locations_abs[cols])
                    for cols in locations_subset
                })

               )


def haversine_distance(data: pd.DataFrame, threshold =1):

        R = 6371 
    
        lat1, lon1, lat2, lon2 = map(np.radians, [data['restaurant_lat'], 
                                              data['restaurant_long'], 
                                              data['delivery_lat'], 
                                              data['delivery_long']])
    
        dlat = lat2 - lat1
        dlon = lon2 - lon1
    
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return (
                data.assign(
                     distance_km = (R * c).round(2)
                    ).assign(
                        distance_km_cat = lambda x: pd.cut(x['distance_km'], bins=[0,5,10,15,21], labels=['short','medium','long','very_long'])
                    )
                )



def extract_time_features(data : pd.DataFrame):

        time_subset = data.loc[:,['order_date','order_time','picked_time']]

        return(
                data.assign(**{
                cols : pd.to_datetime(data[cols].replace("NaN ", np.nan), format="mixed", errors='coerce')
                for cols in time_subset.columns.tolist()
              }).assign(
                 day = lambda x: x['order_date'].dt.day,
                 month = lambda x: x['order_date'].dt.month,
                 day_of_week = lambda x: x['order_date'].dt.day_name(),
                 is_weekend = lambda x: x['order_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int),
                 pickup_time = lambda x: (x['picked_time'] - x['order_time']).dt.seconds.div(60).round(2),
                 order_time_hour = lambda x: x['order_time'].dt.hour
                ).assign(

                        time_of_day = lambda x: pd.cut(x['order_time_hour'], 
                        bins=[1,6,12,17,21,24],labels=['after_midnight','morning',
                        'afternoon','evening','night'],include_lowest=True),
                        pickup_time_cat = lambda x: pd.cut(x['pickup_time'], bins=[0,5,10,15], labels=['5 minutes','10 minutes','15 minutes'],include_lowest=True)
                ).drop(
                    ['id','order_date','order_time','picked_time'],axis=1
                      )
            
            )


def perform_data_cleaning(file_path):

        df = pd.read_csv(file_path)

        cleaned_df = ( df.pipe(column_name_change)
                 .pipe(data_cleaning)
                 .pipe(clean_lat_long)
                 .pipe(haversine_distance)
                 .pipe(extract_time_features)
                 )
    
        return cleaned_df



if __name__ == "__main__":

       df = pd.read_csv(r"C:/Users/User/delivery-time-prediction/data/raw/swiggy.csv")

       print("Dataframe loaded successfully. Starting data cleaning process...")

       perform_data_cleaning(file_path=r"C:/Users/User/delivery-time-prediction/data/raw/swiggy.csv")

       print("Data cleaning process completed successfully.")