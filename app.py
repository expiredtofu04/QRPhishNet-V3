# ============================================================
# QRPhishNet
# Streamlit Application
#
# Dual-Stream QR Phishing Detection
#
# Visual Stream  : QRPhishNet V3 CNN
# URL Stream     : XGBoost
# Explainability : Native XGBoost SHAP contributions
# Fusion         : Weighted Late Fusion
# ============================================================

import streamlit as st
import pandas as pd
from PIL import Image


# ============================================================
# PROJECT UTILITIES
# ============================================================

from utils.cnn_utils import (
    load_cnn_model,
    predict_cnn
)

from utils.preprocessing import (
    preprocess_image
)

from utils.qr_decoder import (
    decode_qr
)

from utils.xgb_utils import (
    load_xgb_model,
    load_feature_columns,
    predict_url
)

from utils.shap_utils import (
    explain_prediction,
    top_features,
    get_feature_contributions,
    get_phishing_features,
    get_genuine_features,
    get_base_value,
    plot_shap_bar
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="QRPhishNet",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .qr-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .qr-subtitle {
        font-size: 18px;
        text-align: center;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        opacity: 0.6;
        margin-top: 50px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

@st.cache_resource
def get_cnn_model():

    return load_cnn_model()


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

@st.cache_resource
def get_xgb_model():

    return load_xgb_model()


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

@st.cache_resource
def get_feature_columns():

    return load_feature_columns()


# ============================================================
# INITIALIZE MODELS
# ============================================================

try:

    cnn_model = get_cnn_model()

except Exception as e:

    st.error(
        f"Unable to load QRPhishNet V3 CNN: {e}"
    )

    st.stop()


try:

    xgb_model = get_xgb_model()

except Exception as e:

    st.error(
        f"Unable to load XGBoost model: {e}"
    )

    st.stop()


try:

    feature_columns = get_feature_columns()

except Exception as e:

    st.error(
        f"Unable to load XGBoost feature columns: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ QRPhishNet")

    st.markdown(
        """
        ### Dual-Stream Detection

        **Visual Stream**

        QRPhishNet V3 CNN

        **URL Stream**

        XGBoost-based URL analysis

        **Explainability**

        Native XGBoost SHAP contributions
        """
    )

    st.divider()

    # ========================================================
    # FUSION SETTINGS
    # ========================================================

    st.subheader("Fusion Settings")

    cnn_weight = st.slider(
        "CNN Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05
    )

    xgb_weight = 1.0 - cnn_weight

    st.write(
        f"XGBoost Weight: **{xgb_weight:.2f}**"
    )

    st.divider()

    # ========================================================
    # RISK THRESHOLDS
    # ========================================================

    st.subheader("Risk Thresholds")

    genuine_threshold = st.slider(
        "Genuine → Suspicious (%)",
        min_value=1,
        max_value=49,
        value=30
    )

    malicious_threshold = st.slider(
        "Suspicious → Malicious (%)",
        min_value=51,
        max_value=99,
        value=70
    )

    st.divider()

    st.caption(
        "QRPhishNet — Dual-Stream QR Phishing Detection"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="qr-title">🔐 QRPhishNet</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="qr-subtitle">
    A Dual-Stream Deep Learning Framework for QR Code
    Phishing Detection
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.info(
    """
    **How QRPhishNet works**

    The system analyzes a QR code through two complementary streams:

    **1. Visual Stream** — QRPhishNet V3 CNN analyzes the
    visual characteristics of the QR image.

    **2. URL Stream** — The embedded URL is decoded and
    analyzed using handcrafted security features and XGBoost.

    **3. Explainability** — Native XGBoost SHAP contributions
    identify which URL features influenced the prediction.

    **4. Weighted Fusion** — CNN and XGBoost probabilities
    are combined using configurable weights.
    """
)


# ============================================================
# QR UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload QR Code</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a QR code image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    help="Upload an image containing a QR code."
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Unable to read image: {e}"
        )

        st.stop()


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.image(
            image,
            caption="Uploaded QR Code",
            width="stretch"
        )


    st.divider()


    # ========================================================
    # SCAN BUTTON
    # ========================================================

    scan_button = st.button(
        "🔍 Scan QR Code",
        type="primary",
        width="stretch"
    )


    # ========================================================
    # SCAN PROCESS
    # ========================================================

    if scan_button:

        # ----------------------------------------------------
        # Clear previous results
        # ----------------------------------------------------

        for key in [
            "cnn_result",
            "decoded_url",
            "xgb_result",
            "shap_values",
            "shap_top",
            "shap_all",
            "shap_phishing",
            "shap_genuine",
            "shap_base_value",
            "fusion_probability",
            "final_risk",
            "scan_started"
        ]:

            st.session_state.pop(
                key,
                None
            )


        # ====================================================
        # CNN INFERENCE
        # ====================================================

        with st.spinner(
            "Running QRPhishNet V3 CNN..."
        ):

            try:

                image_tensor = preprocess_image(
                    image
                )

                cnn_result = predict_cnn(
                    image_tensor,
                    cnn_model
                )

                st.session_state[
                    "cnn_result"
                ] = cnn_result

            except Exception as e:

                st.error(
                    f"CNN prediction failed: {e}"
                )

                st.stop()


        # ====================================================
        # QR DECODING
        # ====================================================

        with st.spinner(
            "Decoding QR code..."
        ):

            try:

                decoded_url = decode_qr(
                    image
                )

                st.session_state[
                    "decoded_url"
                ] = decoded_url

            except Exception as e:

                st.error(
                    f"QR decoding failed: {e}"
                )

                decoded_url = None

                st.session_state[
                    "decoded_url"
                ] = None


        # ====================================================
        # XGBOOST
        # ====================================================

        if decoded_url:

            with st.spinner(
                "Analyzing decoded URL with XGBoost..."
            ):

                try:

                    xgb_result = predict_url(
                        decoded_url,
                        xgb_model,
                        feature_columns
                    )

                    st.session_state[
                        "xgb_result"
                    ] = xgb_result

                except Exception as e:

                    st.error(
                        f"XGBoost prediction failed: {e}"
                    )

                    st.stop()

        else:

            st.warning(
                "Unable to decode a URL from this QR code."
            )

            st.session_state[
                "xgb_result"
            ] = None


        # ====================================================
        # SHAP EXPLAINABILITY
        # ====================================================

        if decoded_url and st.session_state.get(
            "xgb_result"
        ) is not None:

            with st.spinner(
                "Generating SHAP explanation..."
            ):

                try:

                    xgb_result = st.session_state[
                        "xgb_result"
                    ]


                    # ----------------------------------------
                    # Native XGBoost SHAP
                    # ----------------------------------------

                    shap_values = explain_prediction(

                        xgb_model,

                        xgb_result[
                            "feature_vector"
                        ]

                    )


                    # ----------------------------------------
                    # Top features
                    # ----------------------------------------

                    shap_top = top_features(

                        shap_values,

                        xgb_result[
                            "feature_vector"
                        ],

                        top_n=10

                    )


                    # ----------------------------------------
                    # All contributions
                    # ----------------------------------------

                    shap_all = get_feature_contributions(

                        shap_values,

                        xgb_result[
                            "feature_vector"
                        ]

                    )


                    # ----------------------------------------
                    # Phishing-driving features
                    # ----------------------------------------

                    shap_phishing = get_phishing_features(

                        shap_values,

                        xgb_result[
                            "feature_vector"
                        ],

                        top_n=5

                    )


                    # ----------------------------------------
                    # Genuine-driving features
                    # ----------------------------------------

                    shap_genuine = get_genuine_features(

                        shap_values,

                        xgb_result[
                            "feature_vector"
                        ],

                        top_n=5

                    )


                    # ----------------------------------------
                    # Base value
                    # ----------------------------------------

                    shap_base_value = get_base_value(
                        shap_values
                    )


                    # ----------------------------------------
                    # Store SHAP results
                    # ----------------------------------------

                    st.session_state[
                        "shap_values"
                    ] = shap_values

                    st.session_state[
                        "shap_top"
                    ] = shap_top

                    st.session_state[
                        "shap_all"
                    ] = shap_all

                    st.session_state[
                        "shap_phishing"
                    ] = shap_phishing

                    st.session_state[
                        "shap_genuine"
                    ] = shap_genuine

                    st.session_state[
                        "shap_base_value"
                    ] = shap_base_value


                except Exception as e:

                    st.error(
                        f"Unable to generate SHAP explanation: {e}"
                    )


        # ====================================================
        # MARK SCAN COMPLETE
        # ====================================================

        st.session_state[
            "scan_started"
        ] = True


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get(
    "scan_started",
    False
):

    st.markdown(
        '<div class="section-title">🧠 Analysis</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # GET RESULTS
    # ========================================================

    cnn_result = st.session_state.get(
        "cnn_result"
    )

    decoded_url = st.session_state.get(
        "decoded_url"
    )

    xgb_result = st.session_state.get(
        "xgb_result"
    )


    # ========================================================
    # DECODED URL
    # ========================================================

    st.subheader(
        "🔗 Decoded URL"
    )

    if decoded_url:

        st.code(
            decoded_url,
            language="text"
        )

    else:

        st.warning(
            "No URL could be decoded from the QR code."
        )


    # ========================================================
    # TWO STREAMS
    # ========================================================

    col1, col2 = st.columns(2)


    # ========================================================
    # CNN RESULT
    # ========================================================

    with col1:

        st.subheader(
            "🧠 Visual Stream"
        )

        if cnn_result is not None:

            phishing_probability = (
                cnn_result[
                    "phishing_probability"
                ]
            )

            benign_probability = (
                cnn_result[
                    "benign_probability"
                ]
            )


            st.metric(
                "Phishing Probability",
                f"{phishing_probability * 100:.2f}%"
            )


            st.write(
                f"Prediction: **{cnn_result['label']}**"
            )


            st.write(
                f"Genuine Probability: "
                f"**{benign_probability * 100:.2f}%**"
            )


            st.write(
                f"Phishing Probability: "
                f"**{phishing_probability * 100:.2f}%**"
            )


            st.progress(
                float(phishing_probability)
            )

        else:

            st.info(
                "CNN prediction unavailable."
            )


    # ========================================================
    # XGBOOST RESULT
    # ========================================================

    with col2:

        st.subheader(
            "🔗 URL Stream"
        )

        if xgb_result is not None:

            xgb_probability = (
                xgb_result[
                    "phishing_probability"
                ]
            )

            st.metric(
                "XGBoost Phishing Probability",
                f"{xgb_probability * 100:.2f}%"
            )


            st.write(
                f"Prediction: **{xgb_result['label']}**"
            )


            st.write(
                f"Genuine Probability: "
                f"**{xgb_result['genuine_probability'] * 100:.2f}%**"
            )


            st.write(
                f"Phishing Probability: "
                f"**{xgb_probability * 100:.2f}%**"
            )


            st.progress(
                float(xgb_probability)
            )

        else:

            st.info(
                "XGBoost prediction unavailable."
            )


    # ========================================================
    # URL FEATURES
    # ========================================================

    if xgb_result is not None:

        st.divider()

        st.subheader(
            "🔬 Extracted URL Features"
        )


        feature_df = pd.DataFrame({
            "Feature":
                list(
                    xgb_result[
                        "features"
                    ].keys()
                ),

            "Value":
                list(
                    xgb_result[
                        "features"
                    ].values()
                )
        })


        st.dataframe(
            feature_df,
            width="stretch",
            hide_index=True
        )


    # ========================================================
    # WEIGHTED FUSION
    # ========================================================

    st.divider()

    st.subheader(
        "⚖️ Weighted Fusion"
    )


    if (
        cnn_result is not None
        and
        xgb_result is not None
    ):

        cnn_probability = (
            cnn_result[
                "phishing_probability"
            ]
        )

        xgb_probability = (
            xgb_result[
                "phishing_probability"
            ]
        )


        # ----------------------------------------------------
        # Weighted late fusion
        # ----------------------------------------------------

        fusion_probability = (

            cnn_weight *
            cnn_probability

            +

            xgb_weight *
            xgb_probability

        )


        st.session_state[
            "fusion_probability"
        ] = fusion_probability


        st.write(
            f"CNN Weight: **{cnn_weight:.2f}**"
        )

        st.write(
            f"XGBoost Weight: **{xgb_weight:.2f}**"
        )


        st.write(
            f"CNN Probability: "
            f"**{cnn_probability * 100:.2f}%**"
        )

        st.write(
            f"XGBoost Probability: "
            f"**{xgb_probability * 100:.2f}%**"
        )


        st.metric(
            "Fusion Phishing Probability",
            f"{fusion_probability * 100:.2f}%"
        )


        st.progress(
            float(fusion_probability)
        )


    else:

        fusion_probability = None

        st.warning(
            "Fusion requires both CNN and XGBoost predictions."
        )


    # ============================================================
    # FINAL RISK ASSESSMENT
    # ============================================================

    st.divider()

    st.subheader("🛡️ Final Risk Assessment")

    if fusion_probability is not None:

        # --------------------------------------------------------
        # Determine final risk level
        # --------------------------------------------------------

        fusion_percent = fusion_probability * 100

        if fusion_percent < genuine_threshold:

            risk_label = "Genuine"
            risk_icon = "🟢"
            risk_type = "success"

        elif fusion_percent < malicious_threshold:

            risk_label = "Suspicious"
            risk_icon = "🟡"
            risk_type = "warning"

        else:

            risk_label = "Malicious"
            risk_icon = "🔴"
            risk_type = "error"


        # --------------------------------------------------------
        # Risk banner
        # --------------------------------------------------------

        if risk_type == "success":

            st.success(
                f"{risk_icon} {risk_label}"
            )

        elif risk_type == "warning":

            st.warning(
                f"{risk_icon} {risk_label}"
            )

        else:   

            st.error(
                f"{risk_icon} {risk_label}"
            )


        # --------------------------------------------------------
        # Final Risk Score
        # --------------------------------------------------------

        st.write("### Final Risk Score")

        st.metric(
            label="",
            value=f"{fusion_percent:.2f}%"
        )


        # --------------------------------------------------------
        # Risk interpretation
        # --------------------------------------------------------

        if risk_label == "Genuine":

            st.info(
                f"""
                **Risk Level: Genuine**

                The combined model probability is **{fusion_percent:.2f}%**,
                which is below the suspicious threshold of
                **{genuine_threshold}%**.
                """
            )

        elif risk_label == "Suspicious":

            st.warning(
                f"""
                **Risk Level: Suspicious**

                The combined model probability is **{fusion_percent:.2f}%**.
                This falls between the suspicious threshold of
                **{genuine_threshold}%** and the malicious threshold of
                **{malicious_threshold}%**.

                The QR code should be treated with caution.
                """
            )

        else:

            st.error(
                f"""
                **Risk Level: Malicious**

                The combined model probability is **{fusion_percent:.2f}%**,
                exceeding the malicious threshold of
                **{malicious_threshold}%**.

                The QR code is highly likely to be associated with
                phishing or malicious content.
                """
            )

    else:

        st.info(
            "Final risk assessment will appear after both "
            "CNN and XGBoost predictions are available."
        )


    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 SHAP Explainability"
    )


    if (
        "shap_top" in st.session_state
        and
        xgb_result is not None
    ):

        shap_values = st.session_state[
            "shap_values"
        ]

        shap_top = st.session_state[
            "shap_top"
        ]

        shap_phishing = st.session_state[
            "shap_phishing"
        ]

        shap_genuine = st.session_state[
            "shap_genuine"
        ]

        shap_base_value = st.session_state[
            "shap_base_value"
        ]


        st.markdown(
            """
            SHAP explains how individual URL features
            influenced the XGBoost prediction.

            **Positive SHAP values** push the prediction
            toward **Phishing**.

            **Negative SHAP values** push the prediction
            toward **Genuine**.
            """
        )


        # ====================================================
        # TOP FEATURES
        # ====================================================

        st.markdown(
            "### 📊 Top Influential URL Features"
        )


        display_df = shap_top[
            [
                "Feature",
                "Value",
                "SHAP Value",
                "Importance",
                "Direction"
            ]
        ].copy()


        display_df[
            "SHAP Value"
        ] = display_df[
            "SHAP Value"
        ].astype(
            float
        ).round(
            5
        )


        display_df[
            "Importance"
        ] = display_df[
            "Importance"
        ].astype(
            float
        ).round(
            5
        )


        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )


        # ====================================================
        # SHAP BAR CHART
        # ====================================================

        st.markdown(
            "### 📈 Feature Contribution"
        )


        try:

            fig = plot_shap_bar(

                shap_values,

                xgb_result[
                    "feature_vector"
                ],

                top_n=10

            )


            st.pyplot(
                fig,
                width="stretch"
            )

        except Exception as e:

            st.warning(
                f"Unable to display SHAP plot: {e}"
            )


        # ====================================================
        # FEATURE DIRECTION
        # ====================================================

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # PHISHING FEATURES
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "### 🚨 Phishing-Driving Features"
            )


            if not shap_phishing.empty:

                phishing_display = (
                    shap_phishing[
                        [
                            "Feature",
                            "Value",
                            "SHAP"
                        ]
                    ]
                    .copy()
                )


                phishing_display[
                    "SHAP"
                ] = phishing_display[
                    "SHAP"
                ].astype(
                    float
                ).round(
                    5
                )


                st.dataframe(
                    phishing_display,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.success(
                    "No features are pushing "
                    "the prediction toward phishing."
                )


        # ----------------------------------------------------
        # GENUINE FEATURES
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "### ✅ Genuine-Driving Features"
            )


            if not shap_genuine.empty:

                genuine_display = (
                    shap_genuine[
                        [
                            "Feature",
                            "Value",
                            "SHAP"
                        ]
                    ]
                    .copy()
                )


                genuine_display[
                    "SHAP"
                ] = genuine_display[
                    "SHAP"
                ].astype(
                    float
                ).round(
                    5
                )


                st.dataframe(
                    genuine_display,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info(
                    "No features are pushing "
                    "the prediction toward genuine."
                )


        # ====================================================
        # BASE VALUE
        # ====================================================

        st.caption(
            f"XGBoost SHAP base contribution: "
            f"{shap_base_value:.6f}"
        )


    else:

        st.info(
            "SHAP explanation will appear after "
            "a successful XGBoost prediction."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        QRPhishNet • Dual-Stream QR Phishing Detection Framework
        <br>
        Visual Analysis + URL Analysis + Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)