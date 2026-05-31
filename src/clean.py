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
    df_standard['Ambient Light Level'] = df_standard.groupby('Time of Day')['Ambient Light Level'].transform(lambda x: x.fillna(x.median()))
   
    numeric_df = df_standard.select_dtypes(include=['number'])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    df_scaled = pd.DataFrame(scaled_data, columns=numeric_df.columns)

    imputed_results = {}

    for k in [7]:
        imputer = KNNImputer(n_neighbors=k)
        
        # Impute the missing values on the scaled data
        df_imputed_scaled = imputer.fit_transform(df_scaled)
        
        # Un-scale the data back to original sensor values (e.g., actual Humidity %, actual Light levels)
        df_imputed_original = scaler.inverse_transform(df_imputed_scaled)
        
        # Turn back into a clean DataFrame
        df_final = pd.DataFrame(df_imputed_original, columns=numeric_df.columns)
        
        # Optional: If you had text/timestamp columns, glue them back on here:
        # non_numeric = df.select_dtypes(exclude=['number']).reset_index(drop=True)
        # df_final = pd.concat([non_numeric, df_final], axis=1)
        
        # Save to our dictionary so nothing gets overwritten
        imputed_results[k] = df_final

    df_clean = imputed_results[7]
    return df_clean
