import cv2
import numpy as np
from PIL import Image


# ============================================================
# QR CODE DECODER
# ============================================================

def decode_qr(image):
    """
    Decode a QR code from a PIL image using OpenCV.

    Returns:
        str or None:
            Decoded URL if successful, otherwise None.
    """

    # --------------------------------------------------------
    # Convert PIL image to OpenCV format
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        image = np.array(image)

    # --------------------------------------------------------
    # Convert RGB → BGR
    # --------------------------------------------------------

    if len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # --------------------------------------------------------
    # QR detector
    # --------------------------------------------------------

    detector = cv2.QRCodeDetector()

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    data, points, _ = detector.detectAndDecode(
        image
    )

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if data:

        return data.strip()

    return None