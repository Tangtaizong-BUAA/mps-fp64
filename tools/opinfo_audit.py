"""Sample the installed PyTorch OpInfo catalog. Requires test dependency expecttest."""

from contextlib import nullcontext
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import warnings

import torch
from torch.utils._pytree import tree_map, tree_flatten
from torch.testing._internal.common_methods_invocations import op_db


DEVICE_FACTORIES = {
    "arange",
    "linspace",
    "logspace",
    "empty",
    "empty_permuted",
    "empty_strided",
    "eye",
    "ones",
    "zeros",
    "full",
    "scalar_tensor",
}


def target_kwargs(name, kw):
    default_factory = name in DEVICE_FACTORIES or name.startswith("signal.windows.")
    requested = kw.get("device")
    if (default_factory and requested is None) or (
        requested is not None and torch.device(requested).type == "cpu"
    ):
        return dict(kw, device="mps")
    return kw


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("report", type=Path)
    cli.add_argument("--count", type=int, default=1)
    cli.add_argument("--fallback", action="store_true")
    cli.add_argument("--native", action="store_true", help="audit native MPS float32")
    cli.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source or unpacked wheel root",
    )
    cli.add_argument("--only", nargs="+", help="operator names to recheck")
    args = cli.parse_args()
    sys.path.insert(0, str(args.package.resolve()))
    from mps_fp64 import enable

    dtype = torch.float32 if args.native else torch.float64
    warnings.filterwarnings("ignore")
    records = []
    with args.report.open("w") as stream:
        for op in op_db:
            if args.only and op.name not in args.only:
                continue
            row = {"op": op.name, "variant": op.variant_test_name, "checked": 0}
            try:
                if op.name in {
                    "randn",
                    "rand",
                    "normal",
                    "uniform",
                    "randn_like",
                    "rand_like",
                    "randint",
                    "randint_like",
                    "bernoulli",
                    "multinomial",
                    "normal_",
                    "randperm",
                    "random",
                }:
                    row["status"] = "stochastic_separate"
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
                    continue
                torch.manual_seed(1024)
                if dtype not in op.supported_dtypes("cpu"):
                    raise ValueError(f"no CPU {str(dtype).removeprefix('torch.')} sample")
                samples = iter(op.sample_inputs("cpu", dtype, requires_grad=False))
                for _ in range(args.count):
                    sample = next(samples, None)
                    if sample is None:
                        break
                    data = (sample.input, sample.args, sample.kwargs)

                    def transferable(x):
                        return not isinstance(x, torch.Tensor) or (
                            x.layout == torch.strided
                            and not x.is_complex()
                            and x.numel() <= 2**20
                        )

                    if not all(transferable(x) for x in tree_flatten(data)[0]):
                        raise ValueError("sample outside dense real audit")
                    with (
                        nullcontext()
                        if args.native
                        else enable(cpu_fallback=args.fallback)
                    ) as backend:

                        def move(x):
                            if not torch.is_tensor(x):
                                return x
                            # Preserve the sample's backing storage for explicit as_strided offsets.
                            if op.name == "as_strided" and x is sample.input:
                                count = x.untyped_storage().nbytes() // x.element_size()
                                return (
                                    x.as_strided((count,), (1,), 0)
                                    .to("mps")
                                    .as_strided(x.shape, x.stride(), x.storage_offset())
                                )
                            return x.to("mps")

                        gpu, more, kw = tree_map(move, data)
                        # CPU sample generators explicitly target CPU in factory kwargs.
                        kw = target_kwargs(op.name, kw)
                        if (
                            op.name == "tensor_split"
                            and more
                            and torch.is_tensor(more[0])
                        ):
                            more = (sample.args[0], *more[1:])
                        result = op(gpu, *more, **kw)
                        devices = sorted(
                            {
                                str(x.device.type)
                                for x in tree_flatten(result)[0]
                                if torch.is_tensor(x)
                            }
                        )
                        row["output_devices"] = sorted(
                            set(row.get("output_devices", [])) | set(devices)
                        )
                        if any(device != "mps" for device in devices):
                            raise ValueError(f"unexpected output device: {devices}")
                        if not devices:
                            row["host_result_samples"] = (
                                row.get("host_result_samples", 0) + 1
                            )
                        output_stride = (
                            result.stride()
                            if torch.is_tensor(result)
                            and result.layout == torch.strided
                            else None
                        )
                        result = tree_map(
                            lambda x: x.cpu() if torch.is_tensor(x) else x, result
                        )
                    expected = op(sample.input, *sample.args, **sample.kwargs)
                    if op.name in {
                        "empty_like",
                        "new_empty",
                        "new_empty_strided",
                        "empty",
                        "empty_permuted",
                        "empty_strided",
                    }:
                        assert (
                            result.shape == expected.shape
                            and result.dtype == expected.dtype
                            and output_stride == expected.stride()
                        )
                    else:
                        rtol, atol = (3e-4, 3e-5) if args.native else (1e-10, 1e-12)
                        torch.testing.assert_close(
                            result, expected, equal_nan=True, rtol=rtol, atol=atol
                        )
                    row["checked"] += 1
                    if backend is not None and backend.cpu_ops:
                        row.setdefault("cpu_replay", []).extend(backend.cpu_ops)
                row["status"] = "pass" if row["checked"] else "no_sample"
            except Exception as e:
                row["status"] = type(e).__name__
                row["detail"] = str(e)[:350]
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            records.append(row)
    print(dict(Counter(r["status"] for r in records)))


if __name__ == "__main__":
    main()
