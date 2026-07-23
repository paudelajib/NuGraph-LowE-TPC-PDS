#!/usr/bin/env python3

import os
import torch
from collections import defaultdict


CKPTS = {
    "TPC-only": "/home/apaudel/NuGraph/logs/merged1_July3_tpc_only_ep45_bs64/version_0/checkpoints/epoch=44-step=10530.ckpt",
    "TPC+PDS":  "/home/apaudel/NuGraph/logs/merged1_July3_tpc_pds_ep40_bs64/version_0/checkpoints/epoch=44-step=10530.ckpt",
}


DECODER_PATTERNS = {
    "SemanticDecoder": [
        "semantic",
        "semantics",
        "semantic_decoder",
        "semanticdecoder",
    ],
    "FilterDecoder": [
        "filter",
        "filter_decoder",
        "filterdecoder",
    ],
    "EventDecoder": [
        "event",
        "event_decoder",
        "eventdecoder",
        "event_head",
    ],
    "VertexDecoder": [
        "vertex",
        "vertex_decoder",
        "vertexdecoder",
    ],
    "InstanceDecoder": [
        "instance",
        "instance_decoder",
        "instancedecoder",
    ],
    "SpacepointDecoder": [
        "spacepoint",
        "space_point",
        "spacepoint_decoder",
        "spacepointdecoder",
    ],
}


IMPORTANT_HPARAM_WORDS = [
    "head",
    "decoder",
    "event",
    "semantic",
    "filter",
    "vertex",
    "instance",
    "spacepoint",
    "optical",
    "pds",
    "classes",
]


def safe_torch_load(path):
    """
    Handles both older and newer PyTorch versions.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def flatten_hparams(hparams, prefix=""):
    """
    Recursively flatten nested hyperparameter dictionaries.
    """
    flat = {}

    if not isinstance(hparams, dict):
        return flat

    for k, v in hparams.items():
        key = f"{prefix}.{k}" if prefix else str(k)

        if isinstance(v, dict):
            flat.update(flatten_hparams(v, key))
        else:
            flat[key] = v

    return flat


def find_matches(keys, patterns):
    matches = []

    for key in keys:
        kl = key.lower()
        if any(p.lower() in kl for p in patterns):
            matches.append(key)

    return matches


def print_top_level_modules(keys):
    counts = defaultdict(int)

    for k in keys:
        top = k.split(".")[0]
        counts[top] += 1

    print("\n--- Top-level state_dict prefixes ---")
    for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{name:30s} {count:5d}")


def analyze_checkpoint(label, path):
    print("\n" + "=" * 100)
    print(f"{label}")
    print("=" * 100)
    print(f"Checkpoint: {path}")

    if not os.path.exists(path):
        print("\nERROR: checkpoint does not exist.")
        return

    ckpt = safe_torch_load(path)

    print("\nCheckpoint keys:")
    print(list(ckpt.keys()))

    hparams = ckpt.get("hyper_parameters", {})
    flat_hparams = flatten_hparams(hparams)

    print("\n--- Important hyperparameters ---")
    found_any_hparam = False

    for k, v in sorted(flat_hparams.items()):
        kl = k.lower()
        if any(word in kl for word in IMPORTANT_HPARAM_WORDS):
            print(f"{k}: {v}")
            found_any_hparam = True

    if not found_any_hparam:
        print("No obvious decoder/head/event/optical hyperparameters found.")

    state_dict = ckpt.get("state_dict", None)

    if state_dict is None:
        print("\nERROR: No state_dict found in checkpoint.")
        return

    keys = list(state_dict.keys())

    print(f"\nTotal state_dict parameter keys: {len(keys)}")

    print_top_level_modules(keys)

    print("\n--- Decoder activation check from state_dict keys ---")

    summary = {}

    for decoder_name, patterns in DECODER_PATTERNS.items():
        matches = find_matches(keys, patterns)
        summary[decoder_name] = len(matches)

        if len(matches) > 0:
            status = "PRESENT / LIKELY ACTIVE"
        else:
            status = "NOT FOUND"

        print(f"\n{decoder_name:20s}: {len(matches):4d} keys --> {status}")

        for m in matches[:12]:
            print(f"    {m}")

        if len(matches) > 12:
            print(f"    ... {len(matches) - 12} more")

    print("\n--- Optical / PDS check ---")

    optical_patterns = [
        "optical",
        "op",
        "ophit",
        "opflash",
        "pds",
        "pmt",
        "flash",
    ]

    optical_matches = find_matches(keys, optical_patterns)

    print(f"Optical/PDS-like parameter keys: {len(optical_matches)}")

    for m in optical_matches[:20]:
        print(f"    {m}")

    if len(optical_matches) > 20:
        print(f"    ... {len(optical_matches) - 20} more")

    print("\n--- Compact summary ---")

    for decoder_name, nkeys in summary.items():
        if nkeys > 0:
            print(f"{decoder_name:20s}: ACTIVE / PRESENT")
        else:
            print(f"{decoder_name:20s}: not found")

    use_optical = None

    for k, v in flat_hparams.items():
        if k.lower() in ["use_optical", "hparams.use_optical", "model.use_optical"]:
            use_optical = v

    if use_optical is not None:
        print(f"\nuse_optical from hparams: {use_optical}")

    event_classes = None

    for k, v in flat_hparams.items():
        if "event_classes" in k.lower():
            event_classes = v

    if event_classes is not None:
        print(f"event_classes from hparams: {event_classes}")


def main():
    for label, path in CKPTS.items():
        analyze_checkpoint(label, path)


if __name__ == "__main__":
    main()
