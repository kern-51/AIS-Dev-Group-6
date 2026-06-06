
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer 
from sklearn.preprocessing import MinMaxScaler
def data_clean(df):
    df = df.drop_duplicates()
    df_standard = df.round(2)  
    df_standard["HVAC Operation Mode"] = df_standard["HVAC Operation Mode"].str.lower().str.replace(" ", "_")
    df_standard["Activity Level"] = df_standard["Activity Level"].str.replace(" ", "").str.replace("_", "")
  
    string_cols = df_standard.select_dtypes(include=['object', 'category']).columns  
  
    for col in string_cols:
        unique_values = sorted(df_standard[col].dropna().unique())
        mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}  # starts at 1
        # Apply mapping
        df_standard[col] = df_standard[col].map(mapping)

    df_standard = df_standard[
    df_standard['Temperature'].between(-10, 60) &
    df_standard['Humidity'].between(0, 100) &
    (df_standard['CO2_InfraredSensor'] >= 0) &
    (df_standard['CO2_ElectroChemicalSensor'] >= 0)]
    
    df_standard['Ambient Light Level'] = df_standard.groupby('Time of Day')['Ambient Light Level'].transform(lambda x: x.fillna(x.median()))
   
    numeric_df = df_standard.select_dtypes(include=['number'])


    features = [
        'Temperature',
        'Humidity',
        'CO2_InfraredSensor',
        'CO2_ElectroChemicalSensor',
        'MetalOxideSensor_Unit1',
        'MetalOxideSensor_Unit2',
        'MetalOxideSensor_Unit3',
        'MetalOxideSensor_Unit4'
    ]
    
    imputer = KNNImputer(n_neighbors=7)
    
    df_standard[features] = imputer.fit_transform(
        df_standard[features]
    )


    #  Grab ALL numerical columns together so KNN can see the relationships
    numeric_cols = df_standard.select_dtypes(include=["number"]).columns
    # Scale all numerical columns at once
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_standard[numeric_cols])
    # Impute all columns together
    # (KNN will now use Temperature, Humidity, etc., to find the perfect match!)
    imputer = KNNImputer(n_neighbors=7)
    imputed_scaled = imputer.fit_transform(scaled_data)
    # Inverse scale everything back to their original ranges
    imputed_original = scaler.inverse_transform(imputed_scaled)
    # Overwrite the original columns with the completely filled-in data
    df_standard[numeric_cols] = imputed_original
    
    return df_clean
