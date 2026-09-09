# ============================================================
# QRPhishNet
# SHAP Explainability for XGBoost URL Stream
#
# Uses XGBoost native SHAP contributions
# instead of shap.TreeExplainer
# ============================================================

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

def explain_prediction(model, feature_vector):
    """
    Calculate SHAP contributions using XGBoost's
    native pred_contribs=True implementation.

    Returns:
        numpy.ndarray

    Shape:
        (1, number_of_features + 1)

    The final column is the bias/base value.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if feature_vector is None:
        raise ValueError(
            "Feature vector is None."
        )

    if not isinstance(
        feature_vector,
        pd.DataFrame
    ):
        raise TypeError(
            "Feature vector must be a pandas DataFrame."
        )

    if feature_vector.empty:
        raise ValueError(
            "Feature vector is empty."
        )


    # --------------------------------------------------------
    # Copy feature vector
    # --------------------------------------------------------

    X = feature_vector.copy()


    # --------------------------------------------------------
    # Convert all values to numeric
    #
    # Handles values such as:
    # 0.5
    # "0.5"
    # "5E-1"
    # "[5E-1]"
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = X[column].apply(
            _convert_to_numeric
        )


    # --------------------------------------------------------
    # Force float64
    # --------------------------------------------------------

    X = X.astype(
        np.float64
    )


    # --------------------------------------------------------
    # Check invalid values
    # --------------------------------------------------------

    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise ValueError(
            "Feature vector contains "
            "NaN or infinite values."
        )


    # --------------------------------------------------------
    # Get XGBoost booster
    # --------------------------------------------------------

    booster = model.get_booster()


    # --------------------------------------------------------
    # Create DMatrix
    # --------------------------------------------------------

    dmatrix = xgb.DMatrix(
        X,
        feature_names=list(X.columns)
    )


    # --------------------------------------------------------
    # Native XGBoost SHAP
    # --------------------------------------------------------

    shap_values = booster.predict(
        dmatrix,
        pred_contribs=True
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    shap_values = np.asarray(
        shap_values
    )


    # --------------------------------------------------------
    # Validate SHAP dimensions
    # --------------------------------------------------------

    expected_columns = (
        len(X.columns) + 1
    )


    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_values.shape}"
        )


    if shap_values.shape[1] != expected_columns:

        raise ValueError(
            "Unexpected number of SHAP values. "
            f"Expected {expected_columns}, "
            f"received {shap_values.shape[1]}."
        )


    return shap_values


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def _convert_to_numeric(value):
    """
    Safely convert feature values to float.

    Handles:
        0.5
        "0.5"
        "5E-1"
        "[5E-1]"
        "[0.5]"
    """

    # --------------------------------------------------------
    # Already numeric
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):
        return float(value)


    # --------------------------------------------------------
    # Convert to string
    # --------------------------------------------------------

    value = str(value).strip()


    # --------------------------------------------------------
    # Remove brackets
    # --------------------------------------------------------

    if (
        value.startswith("[")
        and
        value.endswith("]")
    ):

        value = value[
            1:-1
        ].strip()


    # --------------------------------------------------------
    # Remove accidental whitespace
    # --------------------------------------------------------

    value = value.strip()


    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    try:

        return float(value)

    except ValueError:

        raise ValueError(
            f"Unable to convert feature value "
            f"'{value}' to float."
        )


# ============================================================
# GET TOP FEATURES
# ============================================================

def top_features(
    shap_values,
    feature_vector,
    top_n=10
):
    """
    Return the most influential URL features.
    """

    feature_names = list(
        feature_vector.columns
    )


    # --------------------------------------------------------
    # Remove final bias value
    # --------------------------------------------------------

    values = shap_values[0][:-1]


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame({

        "Feature":
            feature_names,

        "Value":
            feature_vector.iloc[0].values,

        "SHAP Value":
            values

    })


    # --------------------------------------------------------
    # Absolute importance
    # --------------------------------------------------------

    df["Importance"] = (
        df["SHAP Value"].abs()
    )


    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    df["Direction"] = np.where(

        df["SHAP Value"] > 0,

        "Phishing",

        "Genuine"

    )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(

        "Importance",

        ascending=False

    ).reset_index(
        drop=True
    )


    return df.head(
        top_n
    )


# ============================================================
# ALL FEATURE CONTRIBUTIONS
# ============================================================

def get_feature_contributions(
    shap_values,
    feature_vector
):
    """
    Return SHAP contributions for every feature.
    """

    feature_names = list(
        feature_vector.columns
    )


    values = shap_values[0][:-1]


    df = pd.DataFrame({

        "Feature":
            feature_names,

        "Value":
            feature_vector.iloc[0].values,

        "SHAP":
            values

    })


    df["Absolute_SHAP"] = (
        df["SHAP"].abs()
    )


    df["Direction"] = np.where(

        df["SHAP"] > 0,

        "Phishing",

        "Genuine"

    )


    df = df.sort_values(

        "Absolute_SHAP",

        ascending=False

    ).reset_index(
        drop=True
    )


    return df


# ============================================================
# GET PHISHING-DRIVING FEATURES
# ============================================================

def get_phishing_features(
    shap_values,
    feature_vector,
    top_n=5
):
    """
    Features with positive SHAP contributions.
    These push the prediction toward phishing.
    """

    df = get_feature_contributions(
        shap_values,
        feature_vector
    )


    df = df[
        df["SHAP"] > 0
    ]


    return df.head(
        top_n
    )


# ============================================================
# GET GENUINE-DRIVING FEATURES
# ============================================================

def get_genuine_features(
    shap_values,
    feature_vector,
    top_n=5
):
    """
    Features with negative SHAP contributions.
    These push the prediction toward genuine.
    """

    df = get_feature_contributions(
        shap_values,
        feature_vector
    )


    df = df[
        df["SHAP"] < 0
    ]


    return df.head(
        top_n
    )


# ============================================================
# GET BASE VALUE
# ============================================================

def get_base_value(
    shap_values
):
    """
    Return the XGBoost base/bias contribution.
    """

    return float(
        shap_values[0][-1]
    )


# ============================================================
# PLOT SHAP BAR
# ============================================================

def plot_shap_bar(
    shap_values,
    feature_vector,
    top_n=10
):
    """
    Create a horizontal SHAP contribution plot.

    Positive:
        pushes toward phishing

    Negative:
        pushes toward genuine
    """

    df = get_feature_contributions(

        shap_values,

        feature_vector

    ).head(
        top_n
    )


    # --------------------------------------------------------
    # Reverse order for horizontal plot
    # --------------------------------------------------------

    df = df.iloc[::-1]


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.barh(

        df["Feature"],

        df["SHAP"]

    )


    # --------------------------------------------------------
    # Zero line
    # --------------------------------------------------------

    ax.axvline(
        0,
        linewidth=1
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "SHAP Contribution"
    )

    ax.set_ylabel(
        "URL Feature"
    )

    ax.set_title(
        "Top URL Features Influencing Prediction"
    )


    fig.tight_layout()


    return fig