import concurrent.futures
import unittest
import numpy as np
import torch

from mps_fp64 import enable

F64 = torch.float64


def identical(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    np.testing.assert_array_equal(
        (a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b)), True
    )


class Core(unittest.TestCase):
    def test_binary64_bits(self):
        rng = np.random.default_rng(9039)
        # Random exponent/significand, close exponent cancellation, all corner pairs.
        edge = np.array(
            [
                0,
                1,
                2,
                2**52 - 1,
                2**52,
                2**52 + 1,
                0x3CA0000000000000,
                0x3FF0000000000000,
                0x3FF0000000000001,
                0x7FEFFFFFFFFFFFFF,
                0x7FF0000000000000,
                0x7FF8000000000001,
                0x7FF0000000000001,
            ],
            dtype=np.uint64,
        )
        edge = np.r_[edge, edge ^ np.uint64(1 << 63)]
        a, b = rng.bit_generator.random_raw((2, 200000))
        a, b = (
            np.r_[a, a, edge.repeat(len(edge))],
            np.r_[b, (a ^ np.uint64(1 << 63)) ^ np.uint64(3), np.tile(edge, len(edge))],
        )
        a, b = a.view(np.float64), b.view(np.float64)
        with enable() as mode, np.errstate(all="ignore"):
            x, y = torch.from_numpy(a).to("mps"), torch.from_numpy(b).to("mps")
            for op in (torch.add, torch.sub, torch.mul, torch.div):
                identical(
                    op(x, y).cpu().numpy(),
                    op(torch.from_numpy(a), torch.from_numpy(b)).numpy(),
                )
            identical(x.sqrt().cpu().numpy(), np.sqrt(a))
            self.assertFalse(mode.cpu_ops)

    def test_transcendentals(self):
        rng = np.random.default_rng(42)
        extreme = rng.bit_generator.random_raw(2048).view(np.float64)
        x = np.r_[
            np.linspace(-3, 3, 2049),
            extreme,
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]
        with enable(), np.errstate(all="ignore"):
            g = torch.from_numpy(x).to("mps")
            for op, ref in [
                (torch.sin, np.sin),
                (torch.cos, np.cos),
                (torch.tan, np.tan),
                (torch.exp, np.exp),
                (torch.log, np.log),
                (torch.tanh, np.tanh),
                (torch.log1p, np.log1p),
                (torch.expm1, np.expm1),
                (torch.asin, np.arcsin),
                (torch.atan, np.arctan),
            ]:
                result = op(g).cpu().numpy()
                # Restrict the comparison to 4 representable numbers, also near zero.
                expected = ref(x)
                np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
                valid = ~np.isnan(expected)
                np.testing.assert_array_max_ulp(
                    result[valid], expected[valid], maxulp=4
                )

    def test_graph(self):
        torch.manual_seed(4)
        source = torch.randn(7, 5, dtype=F64)
        weights = torch.randn(5, 4, dtype=F64)

        def run(device):
            x = source.detach().to(device).requires_grad_()
            w = weights.detach().to(device).requires_grad_()
            loss = ((x @ w).tanh().square()).mean()
            loss.backward()
            return loss.detach().cpu(), x.grad.cpu(), w.grad.cpu()

        reference = run("cpu")
        with enable() as mode:
            result = run("mps")
            self.assertFalse(mode.cpu_ops)
        for a, b in zip(result, reference):
            torch.testing.assert_close(a, b, atol=3e-15, rtol=3e-14)

    def test_fp64_residual_and_lu(self):
        # This system becomes singular in FP32.
        eps = 2**-40
        a = torch.tensor([[1.0, 1.0], [1.0, 1.0 + eps]], dtype=F64)
        b = torch.tensor([2.0, 2.0 + eps], dtype=F64)
        with enable():
            result = torch.linalg.solve(a.to("mps"), b.to("mps")).cpu()
            identical(result.numpy(), np.ones(2))
            _, _, info = torch.linalg.lu_factor_ex(
                torch.zeros((4, 4), device="mps", dtype=F64)
            )
            self.assertEqual(info.item(), 1)
            with self.assertRaises(torch.linalg.LinAlgError):
                torch.linalg.solve(
                    torch.zeros((2, 2), device="mps", dtype=F64), b.to("mps")
                )

    def test_views_gather_and_alias(self):
        a = torch.tensor([[1 + 2**-42, -0.0], [3 - 2**-49, 2.0]], dtype=F64)
        with enable():
            x = a.to("mps")
            x.t()[1, 1].fill_(7 + 2**-44)
            a[1, 1] = 7 + 2**-44
            identical(x.cpu().numpy(), a.numpy())
            index = torch.tensor([[1, 0], [0, 1]])
            identical(
                x.gather(1, index.to("mps")).cpu().numpy(), a.gather(1, index).numpy()
            )
            y = torch.zeros((2, 4), device="mps", dtype=F64)
            y[:, ::2].copy_(x)
            identical(y.cpu()[:, ::2].numpy(), a.numpy())
            self.assertEqual(
                torch.ones((0, 3), device="mps", dtype=F64).sum().item(), 0
            )

    def test_threads(self):
        def loop(seed):
            with enable() as mode:
                x = torch.tensor([seed + 2**-42], device="mps", dtype=F64)
                for _ in range(128):
                    x = (x + 2**-44) - 2**-44
                self.assertFalse(mode.cpu_ops)
                return x.item()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            self.assertEqual(
                list(pool.map(loop, [1, 2, 3])), [i + 2**-42 for i in (1, 2, 3)]
            )

    def test_opt_in_cpu_replay(self):
        with enable(cpu_fallback=True) as mode:
            x = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device="mps", dtype=F64)
            ref = torch.linalg.eigvalsh(x.cpu())
            torch.testing.assert_close(torch.linalg.eigvalsh(x).cpu(), ref)
            self.assertTrue(mode.cpu_ops)


if __name__ == "__main__":
    unittest.main()
