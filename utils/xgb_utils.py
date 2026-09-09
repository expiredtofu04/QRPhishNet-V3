import os
import joblib
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MODEL PATHS
# ============================================================

XGB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "qrphishnet_xgboost_v2.joblib"
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.joblib"
)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_xgb_model():

    if not os.path.exists(XGB_MODEL_PATH):

        raise FileNotFoundError(
            f"XGBoost model not found:\n"
            f"{XGB_MODEL_PATH}"
        )

    model = joblib.load(
        XGB_MODEL_PATH
    )

    return model


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

def load_feature_columns():

    if not os.path.exists(FEATURE_COLUMNS_PATH):

        raise FileNotFoundError(
            f"Feature columns file not found:\n"
            f"{FEATURE_COLUMNS_PATH}"
        )

    columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    return columns


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def predict_url(
    url,
    model,
    feature_columns
):

    from utils.feature_extractor import (
        extract_features
    )


    # --------------------------------------------------------
    # Extract URL features
    # --------------------------------------------------------

    features = extract_features(
        url
    )


    # --------------------------------------------------------
    # Convert features to DataFrame
    # --------------------------------------------------------

    feature_vector = pd.DataFrame(
        [features]
    )


    # --------------------------------------------------------
    # Verify feature compatibility
    # --------------------------------------------------------

    missing_features = [
        col
        for col in feature_columns
        if col not in feature_vector.columns
    ]

    extra_features = [
        col
        for col in feature_vector.columns
        if col not in feature_columns
    ]


    # --------------------------------------------------------
    # Missing feature check
    # --------------------------------------------------------

    if missing_features:

        raise ValueError(
            "Missing XGBoost features:\n"
            + "\n".join(
                missing_features
            )
        )


    # --------------------------------------------------------
    # Extra feature check
    # --------------------------------------------------------

    if extra_features:

        raise ValueError(
            "Unexpected XGBoost features:\n"
            + "\n".join(
                extra_features
            )
        )


    # --------------------------------------------------------
    # EXACT TRAINING FEATURE ORDER
    # --------------------------------------------------------

    feature_vector = feature_vector[
        feature_columns
    ]


    # --------------------------------------------------------
    # XGBoost probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        feature_vector
    )[0][1]


    # --------------------------------------------------------
    # Binary prediction
    # --------------------------------------------------------

    prediction = int(
        probability >= 0.5
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if prediction == 1:

        label = "Phishing"

    else:

        label = "Genuine"


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = max(
        probability,
        1 - probability
    )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "label": label,

        "confidence": confidence,

        "phishing_probability": (
            probability
        ),

        "genuine_probability": (
            1 - probability
        ),

        "features": features,

        "feature_vector": feature_vector,

    }