import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Estimator", page_icon="💎", layout="wide")

CUT = ["Ideal", "Premium", "Very Good", "Good", "Fair"]
COLOR = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Newsreader', serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }
.price { font-family: 'Newsreader', serif; font-size: 4.2rem; line-height: 1.05; color: #12303A; }
.result { border-left: 3px solid #B8893B; padding: 0.4rem 0 0.4rem 1.2rem; }
.muted { color: #5B6B73; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "diamond_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error("diamond_model.pkl not found. Run `python train_model.py` first.")
    st.stop()

st.title("Diamond price estimator")
st.markdown(
    "<p class='muted'>Enter the stone's grade and measurements. "
    "The estimate comes from an XGBoost model trained on about 54,000 diamonds.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

with left:
    with st.form("stone"):
        carat = st.slider("Carat (weight)", 0.2, 5.0, 0.70, 0.01)

        a, b, c = st.columns(3)
        cut = a.selectbox("Cut", CUT, help="Ideal is the best cut, Fair the lowest.")
        color = b.selectbox("Color", COLOR, help="D is colorless (best), J has a faint tint.")
        clarity = c.selectbox("Clarity", CLARITY, help="IF is flawless, I1 has visible inclusions.")

        st.markdown("**Measurements**")
        d, e, f = st.columns(3)
        x = d.number_input("Length x (mm)", 3.0, 11.0, 5.70, 0.01)
        y = e.number_input("Width y (mm)", 3.0, 11.0, 5.70, 0.01)
        z = f.number_input("Height z (mm)", 2.0, 7.0, 3.52, 0.01)

        g, h = st.columns(2)
        depth = g.number_input("Depth (%)", 43.0, 79.0, 61.8, 0.1,
                               help="Total depth = z / mean(x, y) × 100")
        table = h.number_input("Table (%)", 43.0, 95.0, 57.0, 0.1,
                               help="Width of the top facet relative to the widest point.")

        submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

# Prediction (runs on first load with defaults, then on every submit)
row = pd.DataFrame([{
    "carat": carat, "cut": cut, "color": color, "clarity": clarity,
    "depth": depth, "table": table, "x": x, "y": y, "z": z,
}])
price_usd = float(model.predict(row)[0])

with right:
    st.subheader("Estimated price")
    currency = st.radio("Currency", ["USD", "INR"], horizontal=True, label_visibility="collapsed")
    if currency == "INR":
        rate = st.number_input("USD to INR rate", 50.0, 150.0, 88.0, 0.5)
        shown = f"₹{price_usd * rate:,.0f}"
    else:
        shown = f"${price_usd:,.0f}"

    st.markdown(f"<div class='result'><div class='price'>{shown}</div></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='muted' style='margin-top:1rem'>{carat:.2f} ct · {cut} cut · "
        f"{color} color · {clarity} clarity</p>",
        unsafe_allow_html=True,
    )

    expected_depth = 2 * z / (x + y) * 100
    if abs(expected_depth - depth) > 3:
        st.warning(
            f"Depth % doesn't match the measurements. x, y and z imply about "
            f"{expected_depth:.1f}%. Check the values you entered."
        )

    st.caption(
        "Prices come from a historical dataset (USD) and are estimates only, "
        "not a valuation or appraisal."
    )