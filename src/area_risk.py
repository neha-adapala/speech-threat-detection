import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


class GeoRiskProvider:
    """
    Production-style geospatial crime risk engine
    for Kaggle London Street Level Crime Data 2024

    Uses:
    - latitude / longitude proximity
    - weighted nearby crime density
    - violent crime boosting
    - dynamic radius queries

    Input:
        lat, lon

    Output:
        low_risk / medium_risk / high_risk
    """

    def __init__(self, data_path="london_crime_2024.csv"):
        self.data_path = data_path
        self.df = None
        self.load_data()

    # ---------------------------------------
    # Load + clean dataset
    # ---------------------------------------
    def load_data(self):
        df = pd.read_csv(self.data_path, low_memory=False)

        required = ["Latitude", "Longitude", "Crime type"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df = df.dropna(subset=["Latitude", "Longitude"])

        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        df = df.dropna(subset=["Latitude", "Longitude"])

        self.df = df.reset_index(drop=True)

    # ---------------------------------------
    # Haversine distance (km)
    # ---------------------------------------
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1))
            * cos(radians(lat2))
            * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    # ---------------------------------------
    # Crime severity weighting
    # ---------------------------------------
    def crime_weight(self, crime_type):
        crime_type = str(crime_type).lower()

        violent_keywords = [
            "violence",
            "robbery",
            "weapon",
            "burglary",
            "theft from person",
            "public order",
            "criminal damage"
        ]

        serious = any(word in crime_type for word in violent_keywords)

        return 2.0 if serious else 1.0

    # ---------------------------------------
    # Main scoring function
    # ---------------------------------------
    def get_risk(self, lat, lon, radius_km=0.5):
        """
        radius_km:
            0.5 = street level
            1.0 = neighbourhood
        """

        total_score = 0
        nearby_count = 0

        for _, row in self.df.iterrows():

            d = self.haversine(
                lat, lon,
                row["Latitude"],
                row["Longitude"]
            )

            if d <= radius_km:

                # closer crimes matter more
                distance_weight = 1 / (d + 0.05)

                severity = self.crime_weight(row["Crime type"])

                total_score += distance_weight * severity
                nearby_count += 1

        # -------------------------
        # Risk bands
        # -------------------------
        if total_score >= 120:
            level = "high_risk"
        elif total_score >= 45:
            level = "medium_risk"
        else:
            level = "low_risk"

        return {
            "risk_level": level,
            "score": round(total_score, 2),
            "nearby_crimes": nearby_count,
            "radius_km": radius_km
        }


# ---------------------------------------
# TEST CASES (Known higher-crime areas)
# ---------------------------------------
if __name__ == "__main__":

    provider = GeoRiskProvider("london_crime_2024.csv")

    tests = {
        "Oxford Circus": (51.5154, -0.1410),
        "Camden Town": (51.5392, -0.1426),
        "Stratford": (51.5413, -0.0032),
        "Croydon": (51.3721, -0.1099),
        "Tower Hamlets": (51.5150, -0.0320),
        "Richmond": (51.4613, -0.3037)
    }

    for name, coords in tests.items():
        result = provider.get_risk(coords[0], coords[1])
        print(name, result)