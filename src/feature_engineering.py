import pandas as pd

def engineer_features(df):
    df = df.copy()

    if "Activity Level" in df.columns:
        df["Activity Level"] = (
            df["Activity Level"]
            .astype(str)
            .str.strip()
            .str.replace("_", " ", regex=False)
        )

        df["Activity Level"] = df["Activity Level"].replace({
            "LowActivity": "Low Activity",
            "ModerateActivity": "Moderate Activity",
            "HighActivity": "High Activity"
        })

    metal_sensor_columns = [
        "MetalOxideSensor_Unit1",
        "MetalOxideSensor_Unit2",
        "MetalOxideSensor_Unit3",
        "MetalOxideSensor_Unit4"
    ]

    df["metal_oxide_avg"] = df[metal_sensor_columns].mean(axis=1)

    df["co2_avg"] = (
        df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]
    ) / 2

    df["co2_sensor_difference"] = (
        df["CO2_ElectroChemicalSensor"] - df["CO2_InfraredSensor"]
    )

    df["gas_intensity_score"] = df[
        ["co2_avg", "CO_GasSensor", "metal_oxide_avg"]
    ].mean(axis=1)

    if "Session ID" in df.columns:
        df.drop(columns=["Session ID"], inplace=True)

    df = pd.get_dummies(
        df,
        columns=[
            "Time of Day",
            "HVAC Operation Mode",
            "Ambient Light Level"
        ],
        drop_first=True
    )

    return df
