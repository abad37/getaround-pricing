🚗 GetAround Pricing Project

📌 Contexte

GetAround est l’Airbnb de la location de voitures.
L’objectif de ce projet est de :

- Analyser les retards de restitution et proposer un seuil minimal entre deux locations (problème produit).
- Optimiser les prix de location avec un modèle de Machine Learning (problème Data Science).
- Mettre en production une API et un Dashboard interactif accessibles en ligne.

🎯 Livrables

1. 📊 Dashboard interactif (Gradio)

👉 https://huggingface.co/spaces/adab82/gradio_getaround

Permet de tester l’API en ligne en renseignant les caractéristiques d’une voiture et en obtenant le prix de location prédit.
Entrée : caractéristiques du véhicule (modèle, kilométrage, carburant, options…)
Sortie : prix prédit en €/jour + niveau de confiance

2. ⚡ API en production (FastAPI)

👉 https://huggingface.co/spaces/adab82/projet_getaround

Endpoints principaux :

POST /predict → prédiction du prix de location

POST /predict_compat → prédiction compatible avec matrices d’entrée

GET /model-info → infos sur le modèle (R², RMSE, MAE, features utilisées)

GET /health → état de santé de l’API

GET /docs → documentation interactive Swagger

Exemple (via curl) :

curl -X POST "https://adab82-projet-getaround.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "model_key": "Citroën",
         "mileage": 50000,
         "engine_power": 120,
         "fuel": "diesel",
         "paint_color": "black",
         "car_type": "hatchback",
         "private_parking_available": true,
         "has_gps": true,
         "has_air_conditioning": true,
         "automatic_car": false,
         "has_getaround_connect": true,
         "has_speed_regulator": true,
         "winter_tires": false
     }'

3. 💻 Code source complet (GitHub)

👉 Repo GitHub

Contient :

- model.py → script pour entraîner le modèle et générer model.pkl
- main.py → code FastAPI (API)
- app.py → dashboard Gradio
- requirements.txt → dépendances
- README.md → documentation du projet

📈 Modèle

Algorithme : Random Forest Regressor
Préprocessing : ColumnTransformer avec OneHotEncoder (catégorielles) et SimpleImputer (numériques)
Dataset : get_around_pricing_project.csv

Performance :

- R² : 0.73
- RMSE : 16.7 €/jour
- MAE : 10.6 €/jour

⚙️ Technologies

- Python 3.9
- Scikit-learn (ML)
- Pandas, Numpy (data)
- Joblib (sérialisation modèle)
- FastAPI (API REST)
- Gradio (dashboard UI)
- Hugging Face Spaces (déploiement cloud)

🚀 Résultats

✔️ Un dashboard interactif utilisable par n’importe quel utilisateur non technique.
✔️ Une API REST en production respectant les standards (Swagger, health-check, endpoints documentés).
✔️ Un modèle reproductible et versionné (via train_model.py et model.pkl).
