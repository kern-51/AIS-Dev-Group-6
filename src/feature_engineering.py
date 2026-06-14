import pandas as pd

def engineer_features(df):
    """
    Performs feature engineering on the cleaned gas monitoring dataset.
    The function standardises Activity Level labels, creates new features
    from existing sensor readings, and removes non-predictive identifiers.
    Engineered features include average metal oxide readings, average CO2
    levels, CO2 sensor differences, and an overall gas intensity score.
    These features provide a more meaningful representation of environmental
    conditions and are intended to improve the performance of downstream
    machine learning models used for activity level prediction.
    """
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

    return df
