# pynuml/labels/flavor_lowe.py
import pandas as pd

class LowEFlavorLabels:
    """
    Two-class flavor labeler for low-energy ν_e studies:
      - cc_nue : charged-current ν_e
      - es_nue : ν_e–e elastic scattering (or any non-CC ν_e in low-E sets)
    Uses both 'is_es' and 'is_cc' flags; 'is_es' takes priority when True.
    """
    def __init__(self):
        self._labels = ("cc_nue", "es_nue")

    @property
    def labels(self): return self._labels

    def label(self, idx: int):
        if not 0 <= idx < len(self._labels):
            raise Exception(f"index {idx} out of range for {len(self._labels)} labels.")
        return self._labels[idx]

    def index(self, name: str) -> int:
        if name not in self._labels:
            raise Exception(f'"{name}" is not the name of a class.')
        return self._labels.index(name)

    @property
    def cc_nue(self): return self.index("cc_nue")
    @property
    def es_nue(self): return self.index("es_nue")

    def __call__(self, event: pd.Series) -> int:
        # tolerate missing/NaN/0/1
        is_cc = bool(event.get("is_cc", 0))
        is_es_val = event.get("is_es", None)
        is_es = (None if pd.isna(is_es_val) else bool(is_es_val))

        # keep ν_e-only assumption for low-E datasets (relax if needed)
        nu_pdg = int(abs(event.get("nu_pdg", 12)))
        if nu_pdg != 12:
            raise Exception(f"Expected nu_pdg==12 (ν_e); got {event.get('nu_pdg')}.")

        if is_es is True:
            return self.es_nue
        if is_cc:
            return self.cc_nue
        # not CC and no explicit ES flag → treat as ES
        return self.es_nue
