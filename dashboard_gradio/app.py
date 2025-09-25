import gradio as gr
import requests
import plotly.graph_objects as go
import json

# =========================
# Config
# =========================
API_BASE   = "https://adab82-projet-getaround.hf.space"  # ⚠️ avec tirets
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL  = f"{API_BASE}/health"
INFO_URL    = f"{API_BASE}/model-info"

DELAY_HEALTH_URL   = f"{API_BASE}/delay/health"
DELAY_KPIS_URL     = f"{API_BASE}/delay/kpis"
DELAY_TRADEOFF_URL = f"{API_BASE}/delay/tradeoff"

# nouveaux endpoints (exposés par l'API)
DELAY_DISTRIB_URL  = f"{API_BASE}/delay/distribution"
DELAY_BYSCOPE_URL  = f"{API_BASE}/delay/by-scope"
DELAY_REVENUE_URL  = f"{API_BASE}/delay/revenue"

# =========================
# Thème & micro-CSS
# =========================
THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0b1220",
    body_text_color="#e5e7eb",
    panel_background_fill="#0f172a",
)

GLOBAL_CSS = """
.gr-box, .gr-panel { border-radius: 14px !important; }
footer {visibility: hidden}
.kpi-badge { background:#1f2937; border-radius:999px; padding:6px 10px; font-weight:600; display:inline-block; }
a { color:#60a5fa !important; }
"""

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

# --------- Helpers d'affichage (KPI & tableaux) ---------
def _kpi_cards(items):
    """
    items: liste de tuples (label, value, suffix)
    -> HTML de cartes KPI
    """
    cards = []
    for label, value, suffix in items:
        cards.append(f"""
        <div style="flex:1; background:#0f172a; color:#e2e8f0; padding:16px; border-radius:12px; 
                    box-shadow:0 2px 8px rgba(0,0,0,.15); text-align:center; min-width:160px;">
            <div style="font-size:12px; opacity:.8; margin-bottom:6px;">{label}</div>
            <div style="font-size:26px; font-weight:700;">{value}{suffix}</div>
        </div>
        """)
    return f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:stretch;">
        {''.join(cards)}
    </div>
    """

def _md_table(rows, headers):
    head = "| " + " | ".join(headers) + " |"
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    return "\n".join([head, sep, body])

def _badges(scope, threshold):
    return f"""
    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:6px;">
      <div class="kpi-badge">Scope : <b>{scope}</b></div>
      <div class="kpi-badge">Seuil sélectionné : <b>{int(threshold)} min</b></div>
      <div class="kpi-badge"><a href="{API_BASE}/docs" target="_blank">OpenAPI /docs</a></div>
    </div>
    """

# --- Badges "dataset OK" pour /delay/health ----
def delay_health_badge():
    try:
        r = requests.get(DELAY_HEALTH_URL, timeout=60)
        data = _safe_json(r)
        if data.get("available"):
            return f"""
            <div class="kpi-badge" style="background:#0b3d2e;">
              Dataset: <b>OK</b> — {data.get('rows', 0)} lignes — <i>{data.get('file','')}</i>
            </div>
            """
        return f"""
        <div class="kpi-badge" style="background:#3d0b0b;">
          Dataset: <b>Indisponible</b> — {data}
        </div>
        """
    except Exception as e:
        return f'<div class="kpi-badge" style="background:#3d0b0b;">Erreur dataset: {e}</div>'

# --- KPIs "propres" pour /delay/kpis ----
def delay_kpis_cards(scope: str):
    try:
        r = requests.get(DELAY_KPIS_URL, params={"scope": scope}, timeout=60)
        data = _safe_json(r)
        k = data.get("kpis", {})

        cards = _kpi_cards([
            ("Taux de retards", round(k.get("retard_%", 0), 2), "%"),
            ("Conflits réels", round(k.get("conflits_réels_%", 0), 2), "%"),
            ("Δ médian avec location précédente", 
             round((k.get("delta_median_avec_prev_min") or 0), 1), " min"),
            ("Retard médian", round((k.get("retard_median_min") or 0), 1), " min"),
        ])
        return cards, data
    except Exception as e:
        return _kpi_cards([("Erreur KPIs", "—", "")]), {"error": str(e)}

# --- Résumé business lisible pour le seuil sélectionné (retour de /delay/tradeoff) ---
def make_trade_summary(current_row: dict, scope: str) -> str:
    if not current_row:
        return "Aucun point pour ce seuil."
    t = current_row
    return (
        f"**Scope :** `{scope}`  •  **Seuil :** **{t['threshold_min']} min**  \n"
        f"**Conflits résolus :** ~**{t.get('conflits_résolus_%_des_conflits', 0)}%**  \n"
        f"**Offres masquées :** ~**{t.get('offres_masquées_%', 0)}%**  \n"
        f"(Conflits réels dans l’échantillon : {t.get('conflits_réels', 0)})"
    )

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

def curl_snippet(model_key, mileage, engine_power, fuel, paint_color, car_type,
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
    return f"""curl -X POST '{PREDICT_URL}' \\
  -H 'Content-Type: application/json' \\
  -d '{json.dumps(payload, ensure_ascii=False)}'"""

def reset_form():
    return ["Citroën", 50000, 120, "diesel", "black", "hatchback",
            True, True, True, False, True, True, False]

# =========================
# Delay Analysis calls
# =========================
def delay_trade(scope, threshold):
    """
    Retourne:
      - data (JSON complet trade-off)
      - fig  (courbe trade-off)
      - current (ligne du seuil courant)
      - reco_text (résumé business / recommandation)
    """
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
        masked = [row.get("offres_masquées_%", 0) for row in rows]
        solved = [row.get("conflits_résolus_%_des_conflits", 0) for row in rows]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=masked, mode="lines+markers", name="% offres masquées"))
        fig.add_trace(go.Scatter(x=xs, y=solved, mode="lines+markers", name="% conflits résolus"))
        fig.update_layout(
            title=f"Trade-off (scope={scope})",
            xaxis_title="Seuil (minutes)",
            yaxis_title="Pourcentage (%)",
            template="plotly_white",
        )

        current = next((row for row in rows if row.get("threshold_min") == int(threshold)), None)

        reco_api = data.get("recommendation")
        if reco_api:
            reco_text = f"**Recommandation API :** {reco_api}"
        else:
            if current:
                reco_text = make_trade_summary(current, scope)
            else:
                reco_text = "Aucune recommandation disponible pour ce seuil."

        return data, fig, current, reco_text

    except Exception as e:
        return {"error": str(e)}, None, None, f"Erreur: {e}"

# ---- Advanced Analysis helpers (affichage joli) ----
def delay_distribution(bins, scope):
    """Histogramme des retards (buckets) + stats clés (KPI cards)"""
    try:
        r = requests.get(DELAY_DISTRIB_URL, params={"bins": int(bins), "scope": scope}, timeout=60)
        data = _safe_json(r)

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

        late_rate = round(data.get("late_rate_%", 0), 2)
        median = round(data.get("median_delay_min", 0), 1)
        p95 = round(data.get("p95_delay_min", 0), 1)

        cards = _kpi_cards([
            ("Taux de retards", late_rate, "%"),
            ("Retard médian", median, " min"),
            ("P95 des retards", p95, " min"),
        ])

        return data, cards, fig

    except Exception as e:
        return {"error": str(e)}, _kpi_cards([("Erreur", "—", "")]), None

def delay_compare_scopes():
    """KPI side-by-side: all vs connect (cartes + tableau + plot)"""
    try:
        r = requests.get(DELAY_BYSCOPE_URL, timeout=60)
        data = _safe_json(r)

        all_ret = round(data["all"]["retard_%"], 2)
        all_conf = round(data["all"]["conflits_réels_%"], 2)
        con_ret = round(data["connect"]["retard_%"], 2)
        con_conf = round(data["connect"]["conflits_réels_%"], 2)

        kpis_html = _kpi_cards([
            ("All • Retards", all_ret, "%"),
            ("All • Conflits réels", all_conf, "%"),
            ("Connect • Retards", con_ret, "%"),
            ("Connect • Conflits réels", con_conf, "%"),
        ])

        labels = ["retard_%", "conflits_réels_%"]
        all_vals = [data["all"][k] for k in labels]
        con_vals = [data["connect"][k] for k in labels]
        fig = go.Figure(data=[
            go.Bar(name='All', x=labels, y=all_vals),
            go.Bar(name='Connect', x=labels, y=con_vals)
        ])
        fig.update_layout(
            barmode='group',
            title="All vs Connect (retards & conflits réels)",
            template="plotly_white",
            yaxis_title="Pourcentage (%)"
        )

        md = _md_table(
            rows=[
                ["All", all_ret, all_conf],
                ["Connect", con_ret, con_conf],
            ],
            headers=["Scope", "Retard (%)", "Conflits réels (%)"]
        )

        return kpis_html, md, fig, data

    except Exception as e:
        return _kpi_cards([("Erreur", "—", "")]), f"Erreur: {e}", None, {"error": str(e)}

def delay_revenue(scope, threshold, avg_price_day, avg_duration_hours):
    """Estimation simple du CA à risque pour un seuil (KPI cards + tableau)"""
    try:
        params = {
            "scope": scope,
            "threshold": int(threshold),
            "avg_price_day": float(avg_price_day),
            "avg_duration_hours": float(avg_duration_hours),
        }
        r = requests.get(DELAY_REVENUE_URL, params=params, timeout=60)
        data = _safe_json(r)

        offres_mask = round(data.get("offres_masquees_%", data.get("offres_masquées_%", 0)), 2)
        revenue_pct = round(data.get("revenue_at_risk_%", 0), 2)
        revenue_abs = round(data.get("revenue_at_risk_abs", 0), 2)
        affected = int(data.get("affected_rentals", 0))

        cards = _kpi_cards([
            ("Offres masquées", offres_mask, "%"),
            ("CA à risque", revenue_pct, "%"),
            ("Impact estimé", revenue_abs, " €"),
            ("Locations affectées", f"{affected}", ""),
        ])

        table_md = _md_table(
            rows=[[scope, int(threshold), offres_mask, revenue_pct, revenue_abs, affected]],
            headers=["Scope", "Seuil (min)", "Offres masquées (%)", "CA à risque (%)", "CA à risque (€)", "Locations affectées"]
        )

        return cards, table_md, data

    except Exception as e:
        return _kpi_cards([("Erreur", "—", "")]), f"Erreur: {e}", {"error": str(e)}

# =========================
# UI
# =========================
with gr.Blocks(title="GetAround — Pricing Dashboard", theme=THEME, css=GLOBAL_CSS) as demo:
    gr.Markdown("# 🚗 GetAround — Pricing Dashboard")

    with gr.Tabs():

        # ---------- Onglet 1 : Pricing ----------
        with gr.Tab("Pricing"):
            gr.Markdown("Test rapide des endpoints de l'API de **pricing**.")

            with gr.Row():
                with gr.Column(scale=1):
                    model_key = gr.Textbox(label="model_key", value="Citroën", placeholder="Ex: Citroën / BMW / Renault")
                    mileage = gr.Number(label="mileage (km)", value=50000, precision=0)
                    engine_power = gr.Number(label="engine_power (ch)", value=120, precision=0)

                    fuel = gr.Dropdown(["diesel","petrol","hybrid","electric"], value="diesel", label="fuel")
                    paint_color = gr.Dropdown(["black","white","grey","blue","red","silver"], value="black", label="paint_color")
                    car_type = gr.Dropdown(
                        ["hatchback","sedan","suv","convertible","van","estate","coupe","mpv"],
                        value="hatchback", label="car_type"
                    )

                    private_parking_available = gr.Checkbox(label="private_parking_available", value=True)
                    has_gps = gr.Checkbox(label="has_gps", value=True)
                    has_air_conditioning = gr.Checkbox(label="has_air_conditioning", value=True)
                    automatic_car = gr.Checkbox(label="automatic_car", value=False)
                    has_getaround_connect = gr.Checkbox(label="has_getaround_connect", value=True)
                    has_speed_regulator = gr.Checkbox(label="has_speed_regulator", value=True)
                    winter_tires = gr.Checkbox(label="winter_tires", value=False)

                    with gr.Row():
                        btn_predict = gr.Button("🚗 Prédire", variant="primary")
                        btn_reset   = gr.Button("↺ Reset")

                    gr.Examples(
                        label="Exemples rapides",
                        examples=[
                            ["Citroën", 50000, 120, "diesel", "black", "hatchback", True, True, True, False, True, True, False],
                            ["BMW", 20000, 190, "petrol", "blue", "sedan", True, True, True, True, True, True, True],
                            ["Renault", 90000, 90, "diesel", "white", "suv", False, False, True, False, False, False, False],
                        ],
                        inputs=[model_key, mileage, engine_power, fuel, paint_color, car_type,
                                private_parking_available, has_gps, has_air_conditioning,
                                automatic_car, has_getaround_connect, has_speed_regulator, winter_tires],
                    )

                with gr.Column(scale=1):
                    out_predict = gr.JSON(label="Réponse /predict")
                    curl_code = gr.Textbox(label="Exemple cURL (copiable)", lines=7, interactive=False)
                    gr.HTML('<div class="kpi-badge">Tip : vous pouvez aussi appeler l’endpoint depuis <a href="/docs" target="_blank">/docs</a></div>')

            btn_predict.click(
                fn=call_api,
                inputs=[model_key, mileage, engine_power, fuel, paint_color, car_type,
                        private_parking_available, has_gps, has_air_conditioning,
                        automatic_car, has_getaround_connect, has_speed_regulator, winter_tires],
                outputs=out_predict
            )
            btn_predict.click(
                fn=curl_snippet,
                inputs=[model_key, mileage, engine_power, fuel, paint_color, car_type,
                        private_parking_available, has_gps, has_air_conditioning,
                        automatic_car, has_getaround_connect, has_speed_regulator, winter_tires],
                outputs=curl_code
            )
            btn_reset.click(
                fn=reset_form, inputs=None,
                outputs=[model_key, mileage, engine_power, fuel, paint_color, car_type,
                         private_parking_available, has_gps, has_air_conditioning,
                         automatic_car, has_getaround_connect, has_speed_regulator, winter_tires]
            )

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

                    btn_dh    = gr.Button("📦 Vérifier le dataset")
                    btn_kpi   = gr.Button("📊 KPIs (scope)")
                    btn_trade = gr.Button("⚖️ Courbe trade-off & recommandation")

                with gr.Column(scale=2):
                    # Bandeau info (dataset + scope/seuil + lien docs)
                    badges_html   = gr.HTML("")

                    # KPIs propres (HTML cards)
                    kpi_cards_html = gr.HTML()

                    # Courbe + résumé business
                    plot_trade = gr.Plot(label="Trade-off")
                    reco_md    = gr.Markdown()

                    # Debug JSON replié
                    with gr.Accordion("🔧 Debug JSON (optionnel)", open=False):
                        box_health = gr.JSON(label="/delay/health")
                        box_kpis   = gr.JSON(label="/delay/kpis")
                        trade_json = gr.JSON(label="/delay/tradeoff (raw)")
                        box_row    = gr.JSON(label="Point courant (raw)")

            # Callbacks Delay Analysis
            btn_dh.click(fn=delay_health_badge, inputs=None, outputs=badges_html)
            btn_dh.click(fn=lambda: _safe_json(requests.get(DELAY_HEALTH_URL, timeout=60)), inputs=None, outputs=box_health)

            btn_kpi.click(fn=lambda sc, th: _badges(sc, th), inputs=[scope, threshold], outputs=badges_html)
            btn_kpi.click(fn=delay_kpis_cards, inputs=scope, outputs=[kpi_cards_html, box_kpis])

            btn_trade.click(fn=lambda sc, th: _badges(sc, th), inputs=[scope, threshold], outputs=badges_html)
            btn_trade.click(
                fn=delay_trade,
                inputs=[scope, threshold],
                outputs=[trade_json, plot_trade, box_row, reco_md]
            )

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
                    # Distribution outputs
                    out_dist_cards = gr.HTML(label="KPIs distribution")
                    out_dist_plot  = gr.Plot(label="Histogramme des retards")
                    with gr.Accordion("JSON brut (debug) – Distribution", open=False):
                        out_dist_json  = gr.JSON(label="Stats distribution (raw)")

                    # Compare outputs
                    out_cmp_cards = gr.HTML(label="KPIs (cartes)")
                    out_cmp_table = gr.Markdown(label="Tableau comparatif")
                    out_cmp_plot  = gr.Plot(label="Comparaison")
                    with gr.Accordion("JSON brut (debug) – Compare", open=False):
                        out_cmp_json = gr.JSON(label="KPIs All vs Connect (raw)")

                    # Revenue outputs
                    out_rev_cards = gr.HTML(label="KPI Revenus à risque")
                    out_rev_table = gr.Markdown(label="Détail calcul")
                    with gr.Accordion("JSON brut (debug) – Revenus", open=False):
                        out_rev_json = gr.JSON(label="Estimation revenus (raw)")

            # Callbacks Advanced
            btn_dist.click(
                fn=delay_distribution,
                inputs=[bins, scope_dist],
                outputs=[out_dist_json, out_dist_cards, out_dist_plot]
            )
            btn_cmp.click(
                fn=delay_compare_scopes,
                inputs=None,
                outputs=[out_cmp_cards, out_cmp_table, out_cmp_plot, out_cmp_json]
            )
            btn_rev.click(
                fn=delay_revenue,
                inputs=[scope_rev, th_rev, avg_price, avg_hours],
                outputs=[out_rev_cards, out_rev_table, out_rev_json]
            )

if __name__ == "__main__":
    demo.launch()
