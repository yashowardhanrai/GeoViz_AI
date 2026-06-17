import streamlit as st
import plotly.express as px

from utils.data_loader import load_data, load_oof_classifier


def show_ai_insights():

    st.title("🤖 AI Insights")

    df = load_data()
    latest = df.iloc[-1]
    risk_score = latest["risk_score"]

    # regime column may be "regime" or "risk_level" depending on pipeline version
    regime = latest.get("regime", latest.get("risk_level", "Unknown"))

    # ============================================
    # CLASSIFIER ALERT (top of page)
    # ============================================

    st.subheader("7-day regime change alert")

    try:
        oof = load_oof_classifier()
        latest_clf = oof.iloc[-1]
        prob = float(latest_clf["pred_cal"])
        threshold = 0.15
        alert_fired = bool(latest_clf["pred_label"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Escalation probability", f"{prob:.3f}")
        col2.metric("Alert threshold", f"{threshold}")
        col3.metric("Alert status", "ACTIVE" if alert_fired else "Clear")

        if alert_fired:
            st.error(
                f"⚠️ Regime change alert active — classifier probability "
                f"({prob:.3f}) exceeds threshold ({threshold}). "
                "Escalation to a higher risk regime likely within 7 days."
            )
        else:
            st.success(
                f"No regime change alert — probability ({prob:.3f}) "
                f"below threshold ({threshold})."
            )

        st.caption(
            "XGBoost Regime Classifier v3 · OOF AUC 0.8106 [0.7403, 0.8780] · "
            "isotonic calibration · threshold optimised for F1"
        )

        if "date" in oof.columns:
            fig_alert = px.scatter(
                oof,
                x="date",
                y="pred_cal",
                color=oof["pred_label"].map({1: "Alert", 0: "No alert"}),
                color_discrete_map={"Alert": "#E24B4A", "No alert": "#888780"},
                title="Classifier alert history (OOF)",
                labels={"pred_cal": "Escalation probability"},
                opacity=0.7,
            )
            fig_alert.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#E24B4A",
                annotation_text=f"Threshold {threshold}",
            )
            st.plotly_chart(fig_alert, use_container_width=True)

    except FileNotFoundError:
        st.warning(
            "oof_predictions_regime_classifier_7d_v3.csv not found. "
            "Run models/regime_classifier.py first."
        )

    st.divider()

    # ============================================
    # CURRENT ASSESSMENT
    # ============================================

    st.subheader("Current assessment")

    col1, col2 = st.columns(2)
    col1.metric("Risk score", round(risk_score, 2))
    col2.metric("Risk regime", regime)

    st.divider()

    # ============================================
    # NARRATIVE INSIGHTS
    # ============================================

    st.subheader("Narrative insights")

    insights = []

    if latest["market_stress"] > df["market_stress"].mean():
        insights.append(
            "📈 Market stress is above historical average, indicating elevated financial volatility."
        )

    if abs(latest["daily_return"]) > 2:
        insights.append(
            "🛢️ Large oil price movements are contributing to geopolitical uncertainty."
        )

    if latest["conflict_pressure"] > df["conflict_pressure"].mean():
        insights.append(
            "⚔️ Conflict pressure remains elevated due to persistent violence and intensity."
        )

    if latest["fatalities"] > df["fatalities"].mean():
        insights.append(
            "Fatalities are above average levels, increasing systemic geopolitical risk."
        )

    if latest["media_pressure"] > df["media_pressure"].mean():
        insights.append(
            "📰 Media attention is unusually high, suggesting increasing global focus on conflict events."
        )

    if abs(latest["avg_tone"]) > 3:
        insights.append(
            "📉 News sentiment is highly negative, reinforcing market anxiety."
        )

    if latest["shock_score"] > 50:
        insights.append(
            "⚡ Shock indicators suggest abnormal changes in conflict and news dynamics."
        )

    if latest["risk_momentum"] > 0:
        insights.append("📊 Short-term risk momentum is increasing.")
    else:
        insights.append("📉 Short-term risk momentum is easing.")

    regime_msgs = {
        "Stable":   "✅ Current environment remains relatively stable.",
        "Elevated": "🟡 Risk conditions are elevated and should be monitored.",
        "High":     "🟠 Geopolitical conditions are stressed and require close attention.",
        "Crisis":   "🔴 Crisis conditions detected. Extreme uncertainty persists.",
    }
    if regime in regime_msgs:
        insights.append(regime_msgs[regime])

    for item in insights:
        st.info(item)

    # ============================================
    # OVERALL OUTLOOK
    # ============================================

    st.subheader("Overall outlook")

    if risk_score < 25:
        st.success("Low geopolitical risk environment.")
    elif risk_score < 50:
        st.warning("Moderate geopolitical risk environment.")
    elif risk_score < 75:
        st.error("High geopolitical risk environment.")
    else:
        st.error("Critical geopolitical environment.")