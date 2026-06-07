
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
  
    custom_orders = {
    'Time of Day': ['morning', 'afternoon', 'evening', 'night'],
    'Activity Level': ['LowActivity','ModerateActivity','HighActivity'],
    'Ambient Light Level': ['very_dim','dim','moderate','bright','very_bright']
    }
    
    legend_data = []
    
    for col in string_cols:
        if col in custom_orders:
            ordered_vals = custom_orders[col]
            # Ensure all existing values are in the order list (or handle missing)
            mapping = {val: idx+1 for idx, val in enumerate(ordered_vals)}
        else:
            # Fallback to alphabetical
            unique_values = sorted(df_standard[col].dropna().unique())
            mapping = {val: idx+1 for idx, val in enumerate(unique_values)}
        
        df_standard[col] = df_standard[col].map(mapping)
        
        for val, code in mapping.items():
            legend_data.append({
                "Column": col,
                "Original_Value": val,
                "Encoded_Value": code
            })

    mask_humidity_bad = ~df_standard['Humidity'].between(0, 100)
    mask_co2_bad = df_standard['CO2_InfraredSensor'] < 0
    mask_combined = mask_humidity_bad | mask_co2_bad
    
    # DataFrame with only the rows you want to keep
    df_before = df_standard
    df_standard = df_standard[~mask_combined].copy()
    df_standard.loc[df_standard['Temperature'] > 100, 'Temperature']*=0.1
    
    
    df_standard['Ambient Light Level'] = df_standard.groupby('Time of Day')['Ambient Light Level'].transform(lambda x: x.fillna(x.median()))
   
    numeric_df = df_standard.select_dtypes(include=['number'])

    #KNN
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
    
    #KNN for CO_GasSensor
    features = [
        'Temperature',
        'Humidity',
        'CO2_InfraredSensor',
        'CO2_ElectroChemicalSensor',
        'MetalOxideSensor_Unit1',
        'MetalOxideSensor_Unit2',
        'MetalOxideSensor_Unit3',
        'MetalOxideSensor_Unit4',
        'CO_GasSensor'
    ]
    
    imputer = KNNImputer(n_neighbors=1)
    
    df_standard[features] = imputer.fit_transform(
        df_standard[features]
    )
    
    return df_clean
