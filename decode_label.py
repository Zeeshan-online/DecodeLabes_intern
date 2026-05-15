"""
=============================================================
 LABEL DECODER — All Encoded Columns & Their Meanings
 Dataset: E-Commerce Orders ML Project
=============================================================

This file explains every encoded column used in the ML models,
and how to decode predictions back to human-readable labels.
=============================================================
"""

# ── HOW TO USE ─────────────────────────────────────────────
# Run this file directly to print all mappings, OR
# import decode_label and call it anywhere in your project.

import pandas as pd

# ──────────────────────────────────────────────────────────
# ENCODING MAPS  (original value → integer code)
# ──────────────────────────────────────────────────────────

PRODUCT_ENCODE = {
    'Chair':   0,
    'Desk':    1,
    'Laptop':  2,
    'Monitor': 3,
    'Phone':   4,
    'Printer': 5,
    'Tablet':  6,
}

PAYMENT_METHOD_ENCODE = {
    'Cash':        0,
    'Credit Card': 1,
    'Debit Card':  2,
    'Gift Card':   3,
    'Online':      4,
}

ORDER_STATUS_ENCODE = {
    'Cancelled': 0,
    'Delivered': 1,
    'Pending':   2,
    'Returned':  3,
    'Shipped':   4,
}

COUPON_CODE_ENCODE = {
    'FREESHIP':  0,
    'NoCoupon':  1,   # was NaN in original data
    'SAVE10':    2,
    'WINTER15':  3,
}

REFERRAL_SOURCE_ENCODE = {
    'Email':     0,
    'Facebook':  1,
    'Google':    2,
    'Instagram': 3,
    'Referral':  4,
}

# ──────────────────────────────────────────────────────────
# DECODING MAPS  (integer code → original value)
# ──────────────────────────────────────────────────────────

PRODUCT_DECODE         = {v: k for k, v in PRODUCT_ENCODE.items()}
PAYMENT_METHOD_DECODE  = {v: k for k, v in PAYMENT_METHOD_ENCODE.items()}
ORDER_STATUS_DECODE    = {v: k for k, v in ORDER_STATUS_ENCODE.items()}
COUPON_CODE_DECODE     = {v: k for k, v in COUPON_CODE_ENCODE.items()}
REFERRAL_SOURCE_DECODE = {v: k for k, v in REFERRAL_SOURCE_ENCODE.items()}

# ──────────────────────────────────────────────────────────
# ML MODEL OUTPUT LABELS
# ──────────────────────────────────────────────────────────

# Model 2 — Classification (IsDelivered)
IS_DELIVERED_DECODE = {
    0: 'Not Delivered',
    1: 'Delivered',
}

# Model 3 — Clustering (CustomerSegment)
CUSTOMER_SEGMENT_DECODE = {
    'High-Value (VIP)':       'Customers with highest total spend (~$1,908 avg). '
                              'Prioritize retention, loyalty rewards, and upselling.',
    'Mid-Value (Regular)':    'Repeat customers with moderate spend (~$888 avg per order). '
                              'Target with cross-sell campaigns.',
    'Low-Value (Occasional)': 'One-time or low-spend customers (~$559 avg). '
                              'Re-engage with discount codes and win-back emails.',
}

# ──────────────────────────────────────────────────────────
# UTILITY FUNCTION — decode any column instantly
# ──────────────────────────────────────────────────────────

_ALL_DECODERS = {
    'Product':         PRODUCT_DECODE,
    'Product_enc':     PRODUCT_DECODE,
    'PaymentMethod':   PAYMENT_METHOD_DECODE,
    'PaymentMethod_enc': PAYMENT_METHOD_DECODE,
    'OrderStatus':     ORDER_STATUS_DECODE,
    'OrderStatus_enc': ORDER_STATUS_DECODE,
    'CouponCode':      COUPON_CODE_DECODE,
    'CouponCode_enc':  COUPON_CODE_DECODE,
    'ReferralSource':  REFERRAL_SOURCE_DECODE,
    'ReferralSource_enc': REFERRAL_SOURCE_DECODE,
    'IsDelivered':     IS_DELIVERED_DECODE,
    'Predicted_IsDelivered': IS_DELIVERED_DECODE,
}

def decode_label(column_name: str, encoded_value: int) -> str:
    """
    Decode a single encoded value back to its original label.

    Parameters
    ----------
    column_name   : str  — column name (e.g. 'Product_enc', 'OrderStatus_enc')
    encoded_value : int  — the integer code to decode

    Returns
    -------
    str — the original human-readable label

    Example
    -------
    >>> decode_label('Product_enc', 2)
    'Laptop'
    >>> decode_label('IsDelivered', 1)
    'Delivered'
    """
    decoder = _ALL_DECODERS.get(column_name)
    if decoder is None:
        raise ValueError(f"No decoder found for column '{column_name}'. "
                         f"Available: {list(_ALL_DECODERS.keys())}")
    result = decoder.get(encoded_value)
    if result is None:
        raise ValueError(f"Value '{encoded_value}' not found in decoder for '{column_name}'.")
    return result


def decode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode ALL encoded columns in a DataFrame automatically.
    Adds a new column '<col>_decoded' for each encoded column found.

    Parameters
    ----------
    df : pd.DataFrame — your DataFrame (from submission.csv or model output)

    Returns
    -------
    pd.DataFrame — same DataFrame with extra decoded columns appended

    Example
    -------
    >>> df = pd.read_csv('submission.csv')
    >>> df_decoded = decode_dataframe(df)
    """
    df = df.copy()
    for col in df.columns:
        if col in _ALL_DECODERS:
            df[col + '_decoded'] = df[col].map(_ALL_DECODERS[col])
    return df


# ──────────────────────────────────────────────────────────
# PRINT REFERENCE TABLE (run this file to see all labels)
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    sections = [
        ('Product',         PRODUCT_ENCODE),
        ('PaymentMethod',   PAYMENT_METHOD_ENCODE),
        ('OrderStatus',     ORDER_STATUS_ENCODE),
        ('CouponCode',      COUPON_CODE_ENCODE),
        ('ReferralSource',  REFERRAL_SOURCE_ENCODE),
        ('IsDelivered (Model 2 Output)', IS_DELIVERED_DECODE),
    ]

    print("\n" + "="*55)
    print("  LABEL DECODER REFERENCE TABLE")
    print("="*55)
    for name, mapping in sections:
        print(f"\n── {name} ──")
        if isinstance(list(mapping.keys())[0], str):
            for label, code in sorted(mapping.items(), key=lambda x: x[1]):
                print(f"   {code}  ←→  {label}")
        else:
            for code, label in sorted(mapping.items()):
                print(f"   {code}  ←→  {label}")

    print("\n── CustomerSegment (Model 3 Output) ──")
    for seg, desc in CUSTOMER_SEGMENT_DECODE.items():
        print(f"\n   '{seg}'")
        print(f"   → {desc}")

    print("\n" + "="*55)
    print("  USAGE EXAMPLES")
    print("="*55)
    print("\n  from decode_label import decode_label, decode_dataframe")
    print("\n  # Decode a single value:")
    print("  decode_label('Product_enc', 2)          # → 'Laptop'")
    print("  decode_label('OrderStatus_enc', 0)      # → 'Cancelled'")
    print("  decode_label('IsDelivered', 1)          # → 'Delivered'")
    print("\n  # Decode an entire DataFrame:")
    print("  import pandas as pd")
    print("  df = pd.read_csv('submission.csv')")
    print("  df_decoded = decode_dataframe(df)")
    print("="*55)