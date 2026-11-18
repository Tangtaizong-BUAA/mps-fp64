"""Repeat the Jacobi SVD residual, Moore–Penrose and derivative measurements."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "tests")]
from mps_fp64 import enable
from test_svd import residuals, sign_invariant_gradients


def matrices(seed):
    generator = torch.Generator().manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator)

    cases = {}
    for shape in (
        (3, 6, 5),
        (2, 5, 7),
        (2, 1),
        (1, 13),
        (33, 33),
        (64, 8),
        (128, 128),
        (8192, 2),
    ):
        cases["random_" + "x".join(map(str, shape))] = normal(shape)
    rank_three = normal((2, 9, 3)) @ normal((2, 3, 6))
    cases["rank3_9x6"] = rank_three
    cases["duplicated"] = torch.cat((rank_three, rank_three[..., :1]), dim=-1)
    cases["zero"] = torch.zeros(6, 4, dtype=torch.float64)
    cases["repeated"] = torch.diag(
        torch.tensor([4.0, 4.0, 1.0, 0.0], dtype=torch.float64)
    )
    cases["graded"] = torch.diag(
        torch.tensor(
            [math.ldexp(1.0, e) for e in (900, -900, -1074)], dtype=torch.float64
        )
    )
    h = torch.arange(1, 13, dtype=torch.float64)
    cases["hilbert12"] = 1 / (h + h[:, None] - 1)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed = 9325
    cases = matrices(seed)
    svd, pinv = {}, {}
    with enable() as backend:
        for name, a in cases.items():
            full = a.shape[-2:] not in ((8192, 2), (128, 128))
            factors = torch.linalg.svd(a.to("mps"), full_matrices=full)
            measured = residuals(a, *(x.cpu() for x in factors))
            assert max(measured.values()) < 5e-13, (name, measured)
            svd[name] = measured
        for name in (
            "random_3x6x5",
            "random_2x5x7",
            "rank3_9x6",
            "duplicated",
            "repeated",
            "zero",
        ):
            a = cases[name]
            b = torch.linalg.pinv(a.to("mps")).cpu()
            pairs = {
                "AXA": (a @ b @ a, a),
                "XAX": (b @ a @ b, b),
                "AX_symmetry": ((a @ b).mT, a @ b),
                "XA_symmetry": ((b @ a).mT, b @ a),
                "CPU_pinv": (b, torch.linalg.pinv(a)),
            }
            measured = {
                key: float((x - y).abs().max()) / max(1.0, float(y.abs().max()))
                for key, (x, y) in pairs.items()
            }
            assert max(measured.values()) < 5e-13, (name, measured)
            pinv[name] = measured
        generator = torch.Generator().manual_seed(82)
        a, w, t = (
            torch.randn(shape, dtype=torch.float64, generator=generator)
            for shape in ((5, 3), (5,), (3,))
        )
        actual = sign_invariant_gradients(
            a.to("mps").requires_grad_(), w.to("mps"), t.to("mps")
        )
        reference = sign_invariant_gradients(a.requires_grad_(), w, t)
        errors = dict(
            zip(
                (
                    "svd_gradient",
                    "svd_second_gradient",
                    "pinv_gradient",
                    "pinv_second_gradient",
                ),
                (
                    float((x.cpu().detach() - y.detach()).abs().max())
                    for x, y in zip(actual, reference)
                ),
            )
        )
        assert max(errors.values()) < 2e-12, errors
        assert not backend.cpu_ops
    report = {
        "device": torch.backends.mps.get_name(),
        "torch": torch.__version__,
        "seed": seed,
        "comparison": "SVD reconstruction and spectrum max abs normalized by largest input entry, orthogonality max abs; pinv max abs normalized by max(1, largest abs reference entry)",
        "svd": svd,
        "pinv": pinv,
        "cpu_fallback_ops": 0,
        "gradients": {
            "seed": 82,
            "shape": [5, 3],
            "metric": "maximum absolute deviation from CPU",
            "errors": errors,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
