import json

import pandas as pd
import streamlit as st
import xgboost as xgb

from features import TIERS, build_features, risk_tier

st.set_page_config(page_title="Fintech New-User Churn Risk", layout="wide")


@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model("model.json")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return booster, meta


booster, meta = load_model()

st.title("Fintech New-User Churn Risk")
st.write(
    "Scores new users of a digital-wallet app by their risk of uninstalling within 30 days, "
    "and groups them into four risk tiers a CRM team can act on."
)
st.caption(
    f"Demo model trained on **synthetic data** (test ROC-AUC {meta['test_roc_auc']}). "
    "Method from: F. Lee & N. Zulkarnain, *Predicting New User Uninstallation in Digital Wallet Applications "
    "with Explainable AI*, IEEE ICIMTech 2026."
)

with st.expander("What should the CSV look like?"):
    st.markdown(
        "Semicolon-separated (`;`) with these columns: `Register Date` (dd/mm/yyyy), `Media Source` "
        "(Organic/Paid), `KYC` (Classic/Preffered), `Click_PN`, `Click_Cards`, `Cash In`, "
        "`Top Up ( Amount)`, `Payment or Not`, `Product Payment`, `Product (Amount)`."
    )
    with open("sample_new_users.csv", "rb") as f:
        st.download_button("Download sample CSV", f, file_name="sample_new_users.csv", mime="text/csv")

left, right = st.columns([3, 1])
uploaded = left.file_uploader("Upload new-user data (CSV)", type=["csv"])
use_sample = right.button("Use sample data", width="stretch")

raw = None
if uploaded is not None:
    raw = pd.read_csv(uploaded, sep=None, engine="python")
elif use_sample:
    raw = pd.read_csv("sample_new_users.csv", sep=";")

if raw is None:
    st.info("Upload a CSV or click **Use sample data** to start.")
    st.stop()

try:
    X = build_features(raw, meta["features"])
except KeyError as e:
    st.error(f"Missing column in the CSV: {e}. Open 'What should the CSV look like?' above.")
    st.stop()

proba = booster.predict(xgb.DMatrix(X, feature_names=meta["features"]))
out = raw.copy()
out["Uninstall_Probability"] = proba.round(4)
out["Risk_Tier"] = risk_tier(proba, meta["tier_cuts"])

counts = out["Risk_Tier"].value_counts().reindex(TIERS, fill_value=0)
cols = st.columns(4)
for col, tier in zip(cols, TIERS):
    col.metric(f"{tier} risk", f"{counts[tier]:,}")
    col.caption(f"{counts[tier] / len(out):.0%} of users")

st.subheader("Users by risk tier")
# number the tiers so the chart keeps the Low -> Very high order instead of sorting alphabetically
chart = pd.DataFrame({"Users": counts.values}, index=[f"{i + 1}. {t}" for i, t in enumerate(TIERS)])
st.bar_chart(chart, x_label="Risk tier", y_label="Users")

st.subheader("Highest-risk users")
st.dataframe(out.sort_values("Uninstall_Probability", ascending=False).head(20), width="stretch")

st.download_button("Download all scores (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name="churn_risk_scores.csv", mime="text/csv")

with st.expander("Suggested actions per tier"):
    st.markdown(
        "- **Very high**: first-transaction incentive plus a push notification within 24 hours; "
        "prioritise users from paid campaigns with zero transactions.\n"
        "- **High**: guided onboarding (how to top up, first payment), reminder on day 3.\n"
        "- **Medium**: feature education via in-app cards.\n"
        "- **Low**: regular lifecycle communication."
    )
