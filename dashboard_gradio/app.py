import gradio as gr
import requests
import plotly.graph_objects as go

# =========================
# Config
# =========================
API_BASE   = "https://adab82-projet-getaround.hf.space"  # ⚠️ tirets
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL  = f"{API_BASE}/health"
INFO_URL    = f"{API_BASE}/model-info"

DELAY_HEALTH_URL   = f"{API_BASE}/delay/health"
DELAY_KPIS_URL     = f"{API_BASE}/delay/kpis"
DELAY_TRADEOFF_URL = f"{API_BASE}/delay/tradeoff"

# nouveaux endpoints (à ajouter côté API, voir plus bas)
DELAY_DISTRIB_URL  = f"{API_BASE}/delay/distribution"
DELAY_BYSCOPE_URL  = f"{API_BASE}/delay/by-scope"
DELAY_REVENUE_URL  = f"{API_BASE}/delay/revenue"

# =========================
# Utils
# =========================
def _safe_json(resp):
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        try:
            return resp.json()
        except Exception:
            pass
    return {
        "status": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "body_preview": resp.text[:800],
    }

# =========================
# Pricing calls
# =========================
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

# =========================
# Delay Analysis calls (existants)
# =========================
def delay_health():
    try:
        r = requests.get(DELAY_HEALTH_URL, timeout=60)
        return _safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def delay_kpis(scope):
    try:
        r = requests.get(DELAY_KPIS_URL, params={"scope": scope}, timeout=60)
        return _safe_json(r)
    except Exception as e:
        return {"error": str(e)}

def delay_trade(scope, threshold):
    try:
        thresholds = ",".join([str(x) for x in [0, 15, 30, 45, 60, 90, 120, 180, 240]])
        r = requests.get(
            DELAY_TRADEOFF_URL,
            params={"scope": scope, "thresholds": thresholds},
            timeout=60,
        )
        data = _safe_json(r)

        rows = data.get("tradeoff", [])
        xs = [row["threshold_min"] for row in rows]
        masked = [row["offres_masquées_%"] for row in rows]
        solved = [row["conflits_résolus_%_des_conflits"] for row in rows]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=masked, mode="lines+markers", name="% offres masquées"))
        fig.add_trace(go.Scatter(x=xs, y=solved, mode="lines+markers", name="% conflits résolus"))
        fig.update_layout(
            title=f"Trade-off (scope={scope})",
            xaxis_title="Seuil (minutes)",
            yaxis_title="Pourcentage (%)",
            template="plotly_white",
        )

        current = next((row for row in rows if row["threshold_min"] == int(threshold)), None)
        return data, fig, current
    except Exception as e:
        return {"error": str(e)}, None, None

# =========================
# Advanced Analysis calls (nouveaux)
# =========================
def delay_distribution(bins, scope):
    """Histogramme des retards (buckets) + stats clés"""
    try:
        r = requests.get(DELAY_DISTRIB_URL, params={"bins": int(bins), "scope": scope}, timeout=60)
        data = _safe_json(r)
        # plot
        centers = data.get("centers", [])
        counts  = data.get("counts", [])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=centers, y=counts, name="Nombre de locations"))
        fig.update_layout(
            title=f"Distribution des retards (scope={scope})",
            xaxis_title="Retard au checkout (min, centre de bin)",
            yaxis_title="Nombre",
            template="plotly_white",
        )
        return data, fig
    except Exception as e:
        return {"error": str(e)}, None

def delay_compare_scopes():
    """KPI side-by-side: all vs connect"""
    try:
        r = requests.get(DELAY_BYSCOPE_URL, timeout=60)
        data = _safe_json(r)
        # petit bar chart comparatif
        labels = ["retard_%", "conflits_réels_%"]
        all_vals = [data["all"][k] for k in labels]
        con_vals = [data["connect"][k] for k in labels]
        fig = go.Figure(data=[
            go.Bar(name='All', x=labels, y=all_vals),
            go.Bar(name='Connect', x=labels, y=con_vals)
        ])
        fig.update_layout(barmode='group', title="All vs Connect (retard & conflits réels)", template="plotly_white")
        return data, fig
    except Exception as e:
        return {"error": str(e)}, None

def delay_revenue(scope, threshold, avg_price_day, avg_duration_hours):
    """Estimation simple du CA à risque pour un seuil"""
    try:
        params = {
            "scope": scope,
            "threshold": int(threshold),
            "avg_price_day": float(avg_price_day),
            "avg_duration_hours": float(avg_duration_hours),
        }
        r = requests.get(DELAY_REVENUE_URL, params=params, timeout=60)
        return _safe_json(r)
    except Exception as e:
        return {"error": str(e)}

# =========================
# UI
# =========================
with gr.Blocks(title="GetAround — Pricing Dashboard") as demo:
    gr.Markdown("# 🚗 GetAround — Pricing Dashboard")

    with gr.Tabs():

        # ---------- Onglet 1 : Pricing ----------
        with gr.Tab("Pricing"):
            gr.Markdown("Test rapide des endpoints de l'API de **pricing**.")

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

        # ---------- Onglet 2 : Delay Analysis ----------
        with gr.Tab("Delay Analysis"):
            gr.Markdown(
                "Analyse des **retards** et étude du **seuil minimum** entre deux locations "
                "pour réduire les conflits avec le prochain check-in."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    scope = gr.Radio(choices=["all", "connect"], value="all", label="Scope")
                    threshold = gr.Slider(0, 240, value=60, step=15, label="Seuil (minutes)")
                    btn_dh = gr.Button("📦 /delay/health")
                    btn_kpi = gr.Button("📊 KPIs (scope)")
                    btn_trade = gr.Button("⚖️ Courbe trade-off")
                with gr.Column(scale=2):
                    box_health = gr.JSON(label="Health Delay Dataset")
                    box_kpis = gr.JSON(label="KPIs (scope)")
                    plot_trade = gr.Plot(label="Trade-off")
                    box_row = gr.JSON(label="Ligne seuil courant")

            btn_dh.click(fn=delay_health, inputs=None, outputs=box_health)
            btn_kpi.click(fn=delay_kpis, inputs=scope, outputs=box_kpis)
            btn_trade.click(fn=delay_trade, inputs=[scope, threshold], outputs=[box_kpis, plot_trade, box_row])

        # ---------- Onglet 3 : Advanced Analysis ----------
        with gr.Tab("Advanced Analysis"):
            gr.Markdown("Analyses supplémentaires pour aller plus loin.")

            with gr.Row():
                with gr.Column(scale=1):
                    # Distribution
                    bins = gr.Slider(10, 80, value=40, step=5, label="Nombre de bins (histogramme)")
                    scope_dist = gr.Radio(choices=["all", "connect"], value="all", label="Scope (histogramme)")
                    btn_dist = gr.Button("📈 Distribution des retards")

                    # Compare scopes
                    btn_cmp = gr.Button("🆚 All vs Connect (retard & conflits)")

                    # Revenue at risk
                    scope_rev = gr.Radio(choices=["all", "connect"], value="all", label="Scope (CA)")
                    th_rev = gr.Slider(0, 240, value=60, step=15, label="Seuil (minutes)")
                    avg_price = gr.Number(value=50, precision=2, label="Prix moyen / jour (€)")
                    avg_hours = gr.Number(value=24, precision=0, label="Durée moyenne (heures)")
                    btn_rev = gr.Button("💶 Estimation revenus à risque")
                with gr.Column(scale=2):
                    out_dist_json = gr.JSON(label="Stats distribution")
                    out_dist_plot = gr.Plot(label="Histogramme des retards")
                    out_cmp_json = gr.JSON(label="KPIs All vs Connect")
                    out_cmp_plot = gr.Plot(label="Comparaison")
                    out_rev_json = gr.JSON(label="Estimation revenus")

            btn_dist.click(fn=delay_distribution, inputs=[bins, scope_dist], outputs=[out_dist_json, out_dist_plot])
            btn_cmp.click(fn=delay_compare_scopes, inputs=None, outputs=[out_cmp_json, out_cmp_plot])
            btn_rev.click(fn=delay_revenue, inputs=[scope_rev, th_rev, avg_price, avg_hours], outputs=out_rev_json)

if __name__ == "__main__":
    demo.launch()
