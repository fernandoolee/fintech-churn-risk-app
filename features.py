"""Shared preprocessing: raw new-user CSV -> model features (same steps as the research notebook)."""
import numpy as np
import pandas as pd

BINARY_MAPS = {
    "Click_PN": {"Click": 1, "No Click": 0},
    "Click_Cards": {"Click Cards": 1, "No Click Cards": 0},
    "Cash_In": {"Top Up": 1, "No Top Up": 0},
    "Payment_or_Not": {"Payment": 1, "No Payment": 0},
}
CATEGORICAL = ["Media_Source", "KYC", "Product_Payment"]
TARGET = "Status_User"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace(r"[\(\)]", "", regex=True))
    return df


def to_amount(x):
    if pd.isna(x) or x == "":
        return 0.0
    if isinstance(x, str):
        x = x.replace(".", "").replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return 0.0


def build_features(raw: pd.DataFrame, feature_names=None) -> pd.DataFrame:
    """Return the model matrix. If feature_names is given, align columns to it (missing -> 0)."""
    df = clean_columns(raw)
    df["Register_Date"] = pd.to_datetime(df["Register_Date"], format="%d/%m/%Y", errors="coerce")
    for col in ("Top_Up__Amount", "Product_Amount"):
        df[col] = df[col].apply(to_amount)
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping).fillna(0)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown")
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

    df["Engagement_Score"] = df["Click_PN"] + df["Click_Cards"] + df["Cash_In"] + df["Payment_or_Not"]
    df["Total_Transaction_Amount"] = df["Top_Up__Amount"] + df["Product_Amount"]
    df["Register_Weekday"] = df["Register_Date"].dt.weekday.fillna(0).astype(int)

    X = df.drop(columns=[c for c in ("Register_Date", TARGET, "User_ID_New_User") if c in df.columns])
    X = X.astype({c: int for c in X.columns if X[c].dtype == bool})
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)
    return X.astype(float)


TIERS = ["Low", "Medium", "High", "Very high"]


def risk_tier(prob: np.ndarray, cuts) -> np.ndarray:
    """cuts = 3 ascending probability thresholds separating the 4 tiers."""
    return np.array(TIERS, dtype=object)[np.digitize(prob, cuts)]
