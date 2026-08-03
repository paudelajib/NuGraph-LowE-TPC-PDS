#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pynuml


def xyz_columns(df, prefix):
    """
    Find x/y/z columns for a vector-like field.

    Tries common names:
      start_position_x, start_position_y, start_position_z
      start_position_0, start_position_1, start_position_2
      start_position.x, start_position.y, start_position.z
    """
    candidates = [
        [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"],
        [f"{prefix}_0", f"{prefix}_1", f"{prefix}_2"],
        [f"{prefix}.x", f"{prefix}.y", f"{prefix}.z"],
        [f"{prefix}[0]", f"{prefix}[1]", f"{prefix}[2]"],
        [f"{prefix}0", f"{prefix}1", f"{prefix}2"],
    ]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    # fallback: any columns beginning with prefix
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols = sorted(cols)
    if len(cols) >= 3:
        return cols[:3]

    return None


def unit_vector(vec, eps=1.0e-8):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)

    if norm < eps or not np.isfinite(norm):
        return None

    return vec / norm


def direction_to_angles(u):
    ux, uy, uz = u

    theta_xz = np.degrees(np.arctan2(ux, uz))
    theta_yz = np.degrees(np.arctan2(uy, uz))
    theta_z = np.degrees(np.arccos(np.clip(uz, -1.0, 1.0)))

    return theta_xz, theta_yz, theta_z


def derive_lepton_direction(particles, edeps, radii_cm=(1.0, 2.0, 3.0, 5.0), min_edep=2):
    """
    Derive outgoing lepton initial direction.

    Priority:
      1. primary e+/e- early truth edeps near start_position
      2. fallback start_position -> end_position chord

    Returns:
      dict with direction info, or None if failed
    """

    required_particle_cols = ["g4_id", "parent_id", "type"]
    for col in required_particle_cols:
        if col not in particles.columns:
            raise KeyError(f"particle_table missing required column: {col}")

    # Primary outgoing electron/positron candidates.
    leptons = particles[
        (particles["type"].abs() == 11)
        & (particles["parent_id"] == 0)
    ].copy()

    if leptons.empty:
        return None

    # If start_process exists, prefer primary particles explicitly marked primary.
    if "start_process" in leptons.columns:
        primary_like = leptons[leptons["start_process"].astype(str) == "primary"]
        if not primary_like.empty:
            leptons = primary_like.copy()

    # If multiple candidates, pick largest scalar momentum.
    if "momentum" in leptons.columns:
        leptons = leptons.sort_values("momentum", ascending=False)

    lepton = leptons.iloc[0]
    lepton_g4 = int(lepton["g4_id"])

    start_cols = xyz_columns(particles, "start_position")
    if start_cols is None:
        raise KeyError("Could not find start_position x/y/z columns in particle_table")

    start = lepton[start_cols].astype(float).values

    if not np.all(np.isfinite(start)):
        return None

    # First choice: early edeps from this lepton only.
    if "g4_id" in edeps.columns and not edeps.empty:
        my_edeps = edeps[edeps["g4_id"] == lepton_g4].copy()

        pos_cols = ["x_position", "y_position", "z_position"]

        if (not my_edeps.empty) and all(c in my_edeps.columns for c in pos_cols):
            pos = my_edeps[pos_cols].astype(float).values

            if "energy" in my_edeps.columns:
                weights = my_edeps["energy"].astype(float).values
            elif "energy_fraction" in my_edeps.columns:
                weights = my_edeps["energy_fraction"].astype(float).values
            else:
                weights = np.ones(len(my_edeps), dtype=float)

            r = np.linalg.norm(pos - start.reshape(1, 3), axis=1)

            good_common = (
                np.all(np.isfinite(pos), axis=1)
                & np.isfinite(weights)
                & (weights > 0)
                & np.isfinite(r)
                & (r > 1.0e-5)
            )

            for radius in radii_cm:
                mask = good_common & (r <= radius)

                if int(mask.sum()) >= min_edep:
                    centroid = np.average(pos[mask], axis=0, weights=weights[mask])
                    u = unit_vector(centroid - start)

                    if u is not None:
                        txz, tyz, tz = direction_to_angles(u)
                        return {
                            "u": u,
                            "theta_xz_deg": txz,
                            "theta_yz_deg": tyz,
                            "theta_z_deg": tz,
                            "source": 1,  # early edep
                            "radius_cm": float(radius),
                            "n_edep": int(mask.sum()),
                            "g4_id": lepton_g4,
                        }

    # Fallback: start -> end chord.
    end_cols = xyz_columns(particles, "end_position")
    if end_cols is not None:
        end = lepton[end_cols].astype(float).values

        if np.all(np.isfinite(end)):
            u = unit_vector(end - start)

            if u is not None:
                txz, tyz, tz = direction_to_angles(u)
                return {
                    "u": u,
                    "theta_xz_deg": txz,
                    "theta_yz_deg": tyz,
                    "theta_z_deg": tz,
                    "source": 2,  # chord fallback
                    "radius_cm": np.nan,
                    "n_edep": 0,
                    "g4_id": lepton_g4,
                }

    return None


def event_table_length(h5):
    if "event_table" not in h5:
        raise KeyError("No event_table group found")

    g = h5["event_table"]

    if not isinstance(g, h5py.Group):
        raise TypeError("event_table is not an HDF5 group; this script expects column datasets")

    keys = list(g.keys())
    if not keys:
        raise RuntimeError("event_table has no datasets")

    # Prefer known event-table columns.
    for key in ["is_cc", "is_es", "nu_pdg", "event_id"]:
        if key in g:
            return len(g[key])

    return len(g[keys[0]])


def write_dataset(group, name, values, dtype=None):
    if name in group:
        del group[name]

    if dtype is None:
        group.create_dataset(name, data=values)
    else:
        group.create_dataset(name, data=values.astype(dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input EVT H5 file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output EVT H5 file. Will not overwrite unless --overwrite is used."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output file"
    )
    parser.add_argument(
        "--radii-cm",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 5.0],
        help="Adaptive early-edep radii in cm"
    )
    parser.add_argument(
        "--min-edep",
        type=int,
        default=2,
        help="Minimum number of lepton edeps required in early segment"
    )
    args = parser.parse_args()

    infile = Path(args.input)
    outfile = Path(args.output)

    if not infile.exists():
        raise FileNotFoundError(infile)

    if outfile.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{outfile} already exists. Use --overwrite or choose a new output name."
            )
        outfile.unlink()

    print("Copying input to output:")
    print("  input: ", infile)
    print("  output:", outfile)
    shutil.copy2(infile, outfile)

    with h5py.File(outfile, "r") as h5:
        n_events = event_table_length(h5)

    lep_dir = np.full((n_events, 3), np.nan, dtype=np.float32)
    theta_xz = np.full(n_events, np.nan, dtype=np.float32)
    theta_yz = np.full(n_events, np.nan, dtype=np.float32)
    theta_z = np.full(n_events, np.nan, dtype=np.float32)
    source = np.zeros(n_events, dtype=np.int16)
    radius_used = np.full(n_events, np.nan, dtype=np.float32)
    n_edep = np.zeros(n_events, dtype=np.int32)
    lep_g4_id = np.full(n_events, -1, dtype=np.int64)

    print("\nReading events and deriving lepton direction...")
    f = pynuml.io.File(str(infile))
    # Tell pynuml which HDF5 groups to load for each event.
    # Without this, build_evt() has no tables and crashes.
    f.add_group("particle_table")
    f.add_group("edep_table")

    n_seen = 0
    n_ok = 0
    n_early = 0
    n_chord = 0
    n_fail = 0

    for i, evt in enumerate(f):
        if i >= n_events:
            print("WARNING: pynuml iterator has more events than event_table length. Stopping.")
            break

        n_seen += 1

        try:
            particles = evt["particle_table"]
            edeps = evt["edep_table"]

            out = derive_lepton_direction(
                particles,
                edeps,
                radii_cm=tuple(args.radii_cm),
                min_edep=args.min_edep,
            )

            if out is None:
                n_fail += 1
                continue

            lep_dir[i, :] = out["u"]
            theta_xz[i] = out["theta_xz_deg"]
            theta_yz[i] = out["theta_yz_deg"]
            theta_z[i] = out["theta_z_deg"]
            source[i] = out["source"]
            radius_used[i] = out["radius_cm"]
            n_edep[i] = out["n_edep"]
            lep_g4_id[i] = out["g4_id"]

            n_ok += 1
            if out["source"] == 1:
                n_early += 1
            elif out["source"] == 2:
                n_chord += 1

        except Exception as e:
            n_fail += 1
            print(f"WARNING: failed event index {i}: {e}")

    print("\nWriting new event_table columns...")

    with h5py.File(outfile, "r+") as h5:
        g = h5["event_table"]

        write_dataset(g, "lep_dir_x", lep_dir[:, 0], dtype=np.float32)
        write_dataset(g, "lep_dir_y", lep_dir[:, 1], dtype=np.float32)
        write_dataset(g, "lep_dir_z", lep_dir[:, 2], dtype=np.float32)

        write_dataset(g, "lep_theta_xz_deg", theta_xz, dtype=np.float32)
        write_dataset(g, "lep_theta_yz_deg", theta_yz, dtype=np.float32)
        write_dataset(g, "lep_theta_z_deg", theta_z, dtype=np.float32)

        write_dataset(g, "lep_dir_source", source, dtype=np.int16)
        write_dataset(g, "lep_dir_radius_cm", radius_used, dtype=np.float32)
        write_dataset(g, "lep_dir_n_edep", n_edep, dtype=np.int32)
        write_dataset(g, "lep_g4_id", lep_g4_id, dtype=np.int64)

    print("\nDone.")
    print("Events in event_table:", n_events)
    print("Events seen:          ", n_seen)
    print("Direction OK:         ", n_ok)
    print("  early edep source:  ", n_early)
    print("  chord fallback:     ", n_chord)
    print("Failed:               ", n_fail)
    print("\nOutput file:")
    print(outfile)
    print("\nMeaning of lep_dir_source:")
    print("  0 = failed / NaN direction")
    print("  1 = early lepton edep direction")
    print("  2 = start-to-end chord fallback")


if __name__ == "__main__":
    main()
