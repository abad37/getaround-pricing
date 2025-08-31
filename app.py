import gradio as gr, requests

API_BASE = "https://adab82-projet-getaround.hf.space" 
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL  = f"{API_BASE}/health"
INFO_URL    = f"{API_BASE}/model-info"

def _safe_json(resp):
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        try:
            return resp.json()
        except Exception:
            pass
    # retour “verbeux” si non-JSON
    txt = resp.text
    return {
        "status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "body_preview": txt[:800]  # on limite l'affichage
    }

def ping_health():
    try:
        r = requests.get(HEALTH_URL, timeout=60)
        return _safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def get_model_info():
    try:
        r = requests.get(INFO_URL, timeout=60)
        return _safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def call_api(model_key, mileage, engine_power, fuel, paint_color, car_type,
             private_parking_available, has_gps, has_air_conditioning,
             automatic_car, has_getaround_connect, has_speed_regulator, winter_tires):
    payload = {
        "model_key": model_key,
        "mileage": int(mileage),
        "engine_power": int(engine_power),
        "fuel": fuel,
        "paint_color": paint_color,
        "car_type": car_type,
        "private_parking_available": bool(private_parking_available),
        "has_gps": bool(has_gps),
        "has_air_conditioning": bool(has_air_conditioning),
        "automatic_car": bool(automatic_car),
        "has_getaround_connect": bool(has_getaround_connect),
        "has_speed_regulator": bool(has_speed_regulator),
        "winter_tires": bool(winter_tires),
    }
    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=60)
        return _safe_json(r)
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(title="GetAround Pricing Dashboard") as demo:
    gr.Markdown("# 🚗 GetAround — Pricing Dashboard")
    gr.Markdown("Test rapide des endpoints de l'API.")

    with gr.Row():
        with gr.Column():
            model_key = gr.Textbox(label="model_key", value="Citroën")
            mileage = gr.Number(label="mileage", value=50000, precision=0)
            engine_power = gr.Number(label="engine_power", value=120, precision=0)
            fuel = gr.Textbox(label="fuel", value="diesel")
            paint_color = gr.Textbox(label="paint_color", value="black")
            car_type = gr.Textbox(label="car_type", value="hatchback")
            private_parking_available = gr.Checkbox(label="private_parking_available", value=True)
            has_gps = gr.Checkbox(label="has_gps", value=True)
            has_air_conditioning = gr.Checkbox(label="has_air_conditioning", value=True)
            automatic_car = gr.Checkbox(label="automatic_car", value=False)
            has_getaround_connect = gr.Checkbox(label="has_getaround_connect", value=True)
            has_speed_regulator = gr.Checkbox(label="has_speed_regulator", value=True)
            winter_tires = gr.Checkbox(label="winter_tires", value=False)

            btn_predict = gr.Button("🚗 Prédire (POST /predict)")
            btn_health  = gr.Button("🩺 Tester /health (GET)")
            btn_info    = gr.Button("ℹ️  /model-info (GET)")

        with gr.Column():
            out_predict = gr.JSON(label="Réponse /predict")
            out_health  = gr.JSON(label="Réponse /health")
            out_info    = gr.JSON(label="Réponse /model-info")

    btn_predict.click(
        fn=call_api,
        inputs=[model_key, mileage, engine_power, fuel, paint_color, car_type,
                private_parking_available, has_gps, has_air_conditioning,
                automatic_car, has_getaround_connect, has_speed_regulator, winter_tires],
        outputs=out_predict
    )
    btn_health.click(fn=ping_health, inputs=None, outputs=out_health)
    btn_info.click(fn=get_model_info, inputs=None, outputs=out_info)

if __name__ == "__main__":
    demo.launch()