# pynuml/labels/lowe.py
# pynuml/labels/lowe.py
import pandas as pd
from .standard import StandardLabels
'''
class StandardLabelsLowE(StandardLabels):
    """
    Reclassify only the *primary* electron (PDG 11, parent_id==0, start_process='primary')
    with momentum < e_thr as 'lowE_electron'. Keeps 'invisible' last.
    """
    def __init__(self, e_thr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self._e_thr = e_thr  # momentum units follow your files (often GeV/c)
        self._labels = [
            'pion','muon','kaon','hadron','shower',
            'lowE_electron',          # new
            'michel','diffuse','invisible'
        ]

    def __call__(self, part: pd.DataFrame):
        labels = super().__call__(part)
        if labels is None or labels.empty:
            return labels

        # columns provided by StandardLabels: type, parent_id, start_process, momentum, semantic_label, instance_label
        is_e        = labels['type'].abs().eq(11)
        is_primary  = (labels['parent_id'] == 0) & (labels['start_process'] == 'primary')
        is_shower   = labels['semantic_label'].eq(self.index('shower'))
        is_low_p    = labels['momentum'] < self._e_thr

        mask = is_e & is_primary & is_shower & is_low_p
        labels.loc[mask, 'semantic_label'] = self.index('lowE_electron')
        return labels


'''
from .standard import StandardLabels
import pandas as pd
class StandardLabelsLowE(StandardLabels):
    """
    Reclassify electrons (PDG=11) below a momentum threshold as 'lowE_electron'.
    Keep 'invisible' as the last label.
    """
    def __init__(self, e_thr: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self._e_thr = e_thr
        # Insert new class; keep order consistent and 'invisible' last
        self._labels = [
            'pion','muon','kaon','hadron','shower',
            'lowE_electron',          # NEW class
            'michel','diffuse','invisible'
        ]

    def __call__(self, part: pd.DataFrame):
        labels = super().__call__(part)
        if labels is None or labels.empty:
            return labels

        # 'labels' already has 'type' (PDG) and 'momentum' from StandardLabels
        is_e   = labels['type'].abs().eq(11)
        is_sh  = labels['semantic_label'].eq(self.index('shower'))  # only split electrons labeled as showers
        is_low = labels['momentum'] < self._e_thr                   # momentum units follow your file (likely GeV)

        labels.loc[is_e & is_sh & is_low, 'semantic_label'] = self.index('lowE_electron')
        return labels
