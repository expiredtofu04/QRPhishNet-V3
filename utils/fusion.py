# ============================================================
# QRPhishNet
# Weighted Late Fusion
# ============================================================


def weighted_fusion(
    cnn_probability,
    xgb_probability,
    cnn_weight=0.40
):
    """
    Weighted late fusion of CNN and XGBoost probabilities.

    CNN:
        Visual analysis

    XGBoost:
        URL-based analysis

    Returns:
        Fusion probability
    """

    # --------------------------------------------------------
    # XGBoost automatically receives the remaining weight
    # --------------------------------------------------------

    xgb_weight = 1.0 - cnn_weight

    # --------------------------------------------------------
    # Weighted fusion
    # --------------------------------------------------------

    fusion_probability = (

        cnn_weight * cnn_probability

        +

        xgb_weight * xgb_probability

    )

    return fusion_probability