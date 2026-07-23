"""
Simple ES/CC labels.

Label convention:
    0 = ES
    1 = CC
"""

import h5py
import numpy as np


class ESCCLabels:
    labels = ["es", "cc"]

    def __init__(self, infile):
        with h5py.File(infile, "r") as h5:
            self.is_es = np.asarray(h5["event_table/is_es"][:]).reshape(-1)
            self.is_cc = np.asarray(h5["event_table/is_cc"][:]).reshape(-1)

        self.i = 0

        print("ESCCLabels loaded:")
        print("  ES:", int(np.sum(self.is_es == 1)))
        print("  CC:", int(np.sum(self.is_cc == 1)))

    def __call__(self, event):
        i = self.i
        self.i += 1

        is_es = int(self.is_es[i])
        is_cc = int(self.is_cc[i])

        if is_es == 1 and is_cc == 0:
            return 0  # ES

        if is_es == 0 and is_cc == 1:
            return 1  # CC

        raise ValueError(
            f"Bad ES/CC label at event row {i}: "
            f"is_es={is_es}, is_cc={is_cc}"
        )
