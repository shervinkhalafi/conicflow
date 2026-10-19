"""Export the EMA flow-map weights of a flow-maps (JAX/Flax) checkpoint to .npz.

The shipped model/two_moons_lsd_15k.npz was made with

    python convert_checkpoint.py two_moons_paper_lsd_15.pkl model/two_moons_lsd_15k.npz

from the checkpoint written after 15k steps by flow-maps/py/launchers/learn.py
with configs/two_moons.py (slot 0, "lsd"). It holds exactly the arrays
run_sweep.load_model reads from the .pkl -- the EMA (0.999) phi_mlp kernels and
biases -- bit-for-bit, so loading it needs numpy only, not jax/flax.

Needs flax to run (the conversion is the one place that reads the .pkl).
"""
import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl")
    ap.add_argument("npz")
    a = ap.parse_args()

    from flax.serialization import msgpack_restore
    with open(a.pkl, "rb") as f:
        raw = msgpack_restore(f.read())
    ema_key = list(raw["ema_params"].keys())[0]
    phi = raw["ema_params"][ema_key]["params"]["flow_map"]["phi_mlp"]

    arrays = {f"{d}/{leaf}": np.asarray(phi[d][leaf]) for d in sorted(phi) for leaf in ("kernel", "bias")}
    arrays["step"] = np.asarray(int(raw.get("step", -1)))
    np.savez(a.npz, **arrays)
    print(f"wrote {a.npz}: step={int(arrays['step'])}, ema={ema_key}, layers={sorted(phi)}")


if __name__ == "__main__":
    main()
