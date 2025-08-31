import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


data = pd.read_csv("get_around_pricing_project.csv")


# 1. Sélection des features

target = "rental_price_per_day"

categorical_columns = ["model_key", "fuel", "paint_color", "car_type"]
numeric_columns = [
    "mileage", "engine_power", "private_parking_available", "has_gps",
    "has_air_conditioning", "automatic_car", "has_getaround_connect",
    "has_speed_regulator", "winter_tires"
]

X = data[categorical_columns + numeric_columns]
y = data[target]


# 2. Preprocessing

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_columns),
        ("num", numeric_transformer, numeric_columns),
    ]
)


# 3. Modèle

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


# 4. Entraînement

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)


# 5. Évaluation

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))   # <- corrige ici
mae = mean_absolute_error(y_test, y_pred)

metrics = {"r2": r2, "rmse": rmse, "mae": mae}

print("R²:", r2)
print("RMSE:", rmse)
print("MAE:", mae)



# 6. Sauvegarde

model_bundle = {
    "model_name": "RandomForestRegressor",
    "pipeline": pipeline,
    "categorical_columns": categorical_columns,
    "numeric_columns": numeric_columns,
    "compat_feature_order": categorical_columns + numeric_columns,
    "metrics": metrics,
}

joblib.dump(model_bundle, "model.pkl")
print("✅ Nouveau model.pkl sauvegardé")
