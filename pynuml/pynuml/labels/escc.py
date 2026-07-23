"""
ES/CC event labels for low-energy DUNE / MARLEY-style samples.

Label convention:
    0 = ES
    1 = CC
"""

import pandas as pd


class ESCCLabels:
    labels = ["es", "cc"]

    def __init__(self, strict: bool = False):
        self.strict = strict

    def _to_scalar(self, x):
        if hasattr(x, "iloc"):
            if len(x) == 0:
                raise ValueError("empty pandas object in ESCCLabels")
            return x.iloc[0]
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                pass
        return x

    def _get(self, event, key, default=None):
        if isinstance(event, pd.DataFrame):
            if key not in event.columns:
                return default
            return self._to_scalar(event[key])

        if isinstance(event, pd.Series):
            if key not in event.index:
                return default
            return self._to_scalar(event[key])

        try:
            return self._to_scalar(event[key])
        except Exception:
            return default

    def __call__(self, event) -> int:
        is_es = self._get(event, "is_es", default=None)
        is_cc = self._get(event, "is_cc", default=None)

        if is_es is not None:
            is_es = int(is_es)

        if is_cc is not None:
            is_cc = int(is_cc)

        if self.strict and (is_es is None or is_cc is None):
            raise ValueError(
                "ESCCLabels(strict=True) requires both event_table/is_es "
                "and event_table/is_cc."
            )

        if is_es is not None and is_cc is not None:
            if is_es == 1 and is_cc == 0:
                return 0
            if is_cc == 1 and is_es == 0:
                return 1
            raise ValueError(
                f"Ambiguous ES/CC label: is_es={is_es}, is_cc={is_cc}. "
                "Expected exactly one of them to be 1."
            )

        if is_cc is not None:
            if is_cc == 1:
                return 1
            if is_cc == 0:
                return 0
            raise ValueError(f"Bad is_cc value: {is_cc}. Expected 0 or 1.")

        if is_es is not None:
            if is_es == 1:
                return 0
            if is_es == 0:
                return 1
            raise ValueError(f"Bad is_es value: {is_es}. Expected 0 or 1.")

        raise ValueError(
            "Could not make ES/CC label. Need event_table/is_es and/or event_table/is_cc."
        )
