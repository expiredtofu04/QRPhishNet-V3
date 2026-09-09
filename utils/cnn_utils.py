import os
import torch
import torch.nn as nn


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# V3 MODEL PATH
# ============================================================

V3_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_qrphishnet_v3.pth"
)


# ============================================================
# QRPhishNet V3 MODEL
# ============================================================

class QRPhishNet(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()


        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

        self.block1 = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

        self.block2 = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        # ----------------------------------------------------
        # BLOCK 3
        # ----------------------------------------------------

        self.block3 = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        # ----------------------------------------------------
        # BLOCK 4
        # ----------------------------------------------------

        self.block4 = nn.Sequential(

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )

        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

def load_cnn_model():

    if not os.path.exists(V3_MODEL_PATH):

        raise FileNotFoundError(
            f"V3 CNN model not found at:\n{V3_MODEL_PATH}"
        )


    model = QRPhishNet(
        num_classes=2
    ).to(DEVICE)


    state_dict = torch.load(
        V3_MODEL_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    return model


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_cnn(
    image_tensor,
    model
):

    model.eval()


    with torch.no_grad():

        outputs = model(
            image_tensor.to(DEVICE)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    benign_probability = (
        probabilities[0][0].item()
    )

    phishing_probability = (
        probabilities[0][1].item()
    )


    predicted_class = (
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )


    if predicted_class == 0:

        label = "Genuine"

    else:

        label = "Phishing"


    confidence = max(
        benign_probability,
        phishing_probability
    )


    return {

        "label": label,

        "confidence": confidence,

        "benign_probability":
            benign_probability,

        "phishing_probability":
            phishing_probability

    }