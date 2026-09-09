# ============================================================
# QRPhishNet
# URL Feature Extraction
#
# IMPORTANT:
# This implementation matches the feature extraction
# used during XGBoost training.
# ============================================================

import re
import math
import numpy as np

from urllib.parse import urlparse


# ============================================================
# SUSPICIOUS KEYWORDS
# ============================================================

SUSPICIOUS_KEYWORDS = [

    "login",
    "signin",
    "sign-in",

    "verify",
    "verification",

    "account",
    "update",

    "secure",
    "security",

    "confirm",
    "confirmation",

    "password",
    "passwd",

    "credential",

    "authenticate",
    "authentication",

    "bank",
    "banking",

    "payment",
    "wallet",

    "recover",
    "recovery",

    "unlock",

    "suspend",
    "suspended",

    "alert",

    "invoice",
    "billing",

    "webscr",

    "bonus",
    "gift",
    "claim",
    "free"
]


# ============================================================
# POPULAR BRANDS
#
# IMPORTANT:
# This list MUST match the list used during XGBoost training.
# Replace this list if your training notebook used a different
# POPULAR_BRANDS list.
# ============================================================

POPULAR_BRANDS = [

    "paypal",
    "amazon",
    "apple",
    "microsoft",
    "google",
    "facebook",
    "instagram",
    "netflix",
    "linkedin",
    "twitter",
    "whatsapp",
    "bankofamerica",
    "wellsfargo",
    "chase",
    "citibank",
    "hsbc",
    "mastercard",
    "visa"
]


# ============================================================
# ENTROPY
# ============================================================

def entropy(text):

    if not text:

        return 0.0

    probabilities = [

        text.count(char) / len(text)

        for char in set(text)

    ]

    return -sum(

        p * math.log2(p)

        for p in probabilities

        if p > 0

    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(url):

    # --------------------------------------------------------
    # Parse URL
    # --------------------------------------------------------

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    path = parsed.path.lower()

    query = parsed.query.lower()


    # --------------------------------------------------------
    # Special characters
    # --------------------------------------------------------

    special_chars = re.findall(
        r'[^A-Za-z0-9]',
        url
    )


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    hostname_tokens = re.split(
        r"[.-]",
        domain
    )

    path_tokens = re.split(
        r"[/-]",
        path
    )


    # ========================================================
    # KEYWORD FEATURES
    # ========================================================

    keyword_domain = sum(

        word in domain

        for word in SUSPICIOUS_KEYWORDS

    )


    keyword_path = sum(

        word in path

        for word in SUSPICIOUS_KEYWORDS

    )


    keyword_query = sum(

        word in query

        for word in SUSPICIOUS_KEYWORDS

    )


    # ========================================================
    # BRAND IMPERSONATION
    # ========================================================

    brand_impersonation = int(

        any(

            brand in domain

            for brand in POPULAR_BRANDS

        )

    )


    # ========================================================
    # HOSTNAME TOKEN FEATURES
    # ========================================================

    longest_token = max(

        [
            len(token)
            for token in hostname_tokens
        ]

        if hostname_tokens

        else [0]

    )


    avg_token_length = np.mean(

        [
            len(token)
            for token in hostname_tokens
        ]

        if hostname_tokens

        else [0]

    )


    # ========================================================
    # QUERY PARAMETER COUNT
    # ========================================================

    query_parameters = (

        len(
            query.split("&")
        )

        if query

        else 0

    )


    # ========================================================
    # CONSECUTIVE DIGITS
    # ========================================================

    consecutive_digits = max(

        [
            len(match)

            for match in re.findall(
                r"\d+",
                url
            )
        ]

        or [0]

    )


    # ========================================================
    # CONSECUTIVE SPECIAL CHARACTERS
    # ========================================================

    consecutive_special = max(

        [
            len(match)

            for match in re.findall(
                r"[^A-Za-z0-9]+",
                url
            )
        ]

        or [0]

    )


    # ========================================================
    # FEATURE DICTIONARY
    # ========================================================

    features = {

        # ----------------------------------------------------
        # LENGTH FEATURES
        # ----------------------------------------------------

        "url_length":
            len(url),

        "domain_length":
            len(domain),

        "path_length":
            len(path),

        "query_length":
            len(query),


        # ----------------------------------------------------
        # CHARACTER COUNT FEATURES
        # ----------------------------------------------------

        "num_dots":
            url.count("."),

        "num_hyphens":
            url.count("-"),

        "num_underscores":
            url.count("_"),

        "num_slashes":
            url.count("/"),

        "num_digits":
            sum(
                c.isdigit()
                for c in url
            ),

        "num_letters":
            sum(
                c.isalpha()
                for c in url
            ),

        "num_equals":
            url.count("="),

        "num_question":
            url.count("?"),

        "num_ampersand":
            url.count("&"),

        "num_at":
            url.count("@"),

        "num_percent":
            url.count("%"),


        # ----------------------------------------------------
        # SECURITY FEATURES
        # ----------------------------------------------------

        "https":
            int(
                parsed.scheme == "https"
            ),

        "has_ip":
            int(

                bool(

                    re.search(
                        r"\d+\.\d+\.\d+\.\d+",
                        domain
                    )

                )

            ),

        "port_present":
            int(
                parsed.port is not None
            ),


        # ----------------------------------------------------
        # URL STRUCTURE
        # ----------------------------------------------------

        "subdomain_count":
            max(
                domain.count(".") - 1,
                0
            ),

        "url_depth":
            path.count("/"),


        # ----------------------------------------------------
        # RATIOS
        # ----------------------------------------------------

        "digit_ratio":

            sum(
                c.isdigit()
                for c in url
            )
            /
            max(
                len(url),
                1
            ),

        "special_ratio":

            len(special_chars)
            /
            max(
                len(url),
                1
            ),


        # ----------------------------------------------------
        # ENTROPY
        # ----------------------------------------------------

        "url_entropy":
            entropy(url),

        "host_entropy":
            entropy(domain),

        "path_entropy":
            entropy(path),


        # ----------------------------------------------------
        # KEYWORD FEATURES
        # ----------------------------------------------------

        "keywords_domain":
            keyword_domain,

        "keywords_path":
            keyword_path,

        "keywords_query":
            keyword_query,


        # ----------------------------------------------------
        # URL SHORTENER
        # ----------------------------------------------------

        "shortened_url":

            int(

                any(

                    short in domain

                    for short in [

                        "bit.ly",
                        "tinyurl",
                        "goo.gl",
                        "t.co",
                        "ow.ly",
                        "is.gd",
                        "buff.ly",
                        "adf.ly",
                        "cutt.ly",
                        "rb.gy"

                    ]

                )

            ),


        # ----------------------------------------------------
        # TLD
        # ----------------------------------------------------

        "tld_length":

            len(
                domain.split(".")[-1]
            )

            if "." in domain

            else 0,


        # ----------------------------------------------------
        # HOSTNAME TOKEN FEATURES
        # ----------------------------------------------------

        "longest_hostname_token":
            longest_token,

        "avg_hostname_token":
            avg_token_length,


        # ----------------------------------------------------
        # QUERY FEATURES
        # ----------------------------------------------------

        "query_parameter_count":
            query_parameters,


        # ----------------------------------------------------
        # CHARACTER SEQUENCE FEATURES
        # ----------------------------------------------------

        "max_consecutive_digits":
            consecutive_digits,

        "max_consecutive_special":
            consecutive_special,


        # ----------------------------------------------------
        # BRAND IMPERSONATION
        # ----------------------------------------------------

        "brand_impersonation":
            brand_impersonation

    }


    return features