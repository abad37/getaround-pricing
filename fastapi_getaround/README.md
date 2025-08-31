---
title: GetAround Pricing API
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
tags: [fastapi, machine-learning, pricing, rental, car]
---

# 🚗 GetAround Pricing API

API de prédiction des prix de location **GetAround**, déployée sur Hugging Face avec **FastAPI**.

## 🌍 URL
➡️ [https://adab82-projet_getaround.hf.space](https://adab82-projet_getaround.hf.space)

## 🔑 Endpoints

- `POST /predict` — prédiction riche (features nommées)  
- `POST /predict_compat` — format énoncé (`{"input":[[...]]}`)  
- `GET /model-info` — infos modèle (R², RMSE, MAE)  
- `GET /health` — état API  
- `GET /docs` — Swagger UI  
- `GET /redoc` — ReDoc  

## Exemple (curl)

```bash
curl -X POST "https://adab82-projet_getaround.hf.space/predict" \
 -H "Content-Type: application/json" \
 -d '{"model_key":"Citroën","mileage":50000,"engine_power":120,"fuel":"diesel","paint_color":"black","car_type":"hatchback","private_parking_available":true,"has_gps":true,"has_air_conditioning":true,"automatic_car":false,"has_getaround_connect":true,"has_speed_regulator":true,"winter_tires":false}'
