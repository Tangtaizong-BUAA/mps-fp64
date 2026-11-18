import itertools
import unittest

import torch

from mps_fp64 import enable

F64 = torch.float64


class Histograms(unittest.TestCase):
    def test_edges_neighbors_nonfinite_and_histc(self):
        lo, hi, count = -1.0, 1.0, 10
        edges = torch.linspace(lo, hi, count + 1, dtype=F64)
        low = torch.nextafter(edges, torch.full_like(edges, -float("inf")))
        high = torch.nextafter(edges, torch.full_like(edges, float("inf")))
        samples = torch.cat((low, edges, high, torch.tensor([float("nan"), -float("inf"), float("inf")], dtype=F64)))
        with enable() as mode:
            x = samples.to("mps")
            for arg, refarg, options in [(edges.to("mps"), edges, {}),
                                        (count, count, {"range": (lo, hi)})]:
                hist, returned_edges = torch.histogram(x, arg, **options)
                ref = torch.histogram(samples, refarg, **options)
                self.assertTrue(torch.equal(hist.cpu(), ref.hist))
                self.assertTrue(torch.equal(returned_edges.cpu().view(torch.int64), ref.bin_edges.view(torch.int64)))
            c = torch.histc(x, count, lo, hi).cpu()
            self.assertTrue(torch.equal(c, torch.histc(samples, count, lo, hi)))
            self.assertFalse(torch.equal(c, hist.cpu()))
            # Each point exactly equal to the final edge goes in the last bin.
            repeated = torch.tensor([-float("inf"), 0., 0., 0.5, 1., float("inf")], dtype=F64)
            expected = torch.histogram(samples, repeated)
            result = torch.histogram(x, repeated.to("mps"))
            self.assertTrue(torch.equal(result.hist.cpu(), expected.hist))
            self.assertFalse(mode.cpu_ops)

    def test_weighted_precision_density_and_dd(self):
        # Lose neither small FP64 residuals nor individual nearest-even steps.
        xs = torch.tensor([[.5, .5], [.5, .5], [.5, .5], [2., .5],
                           [1., 1.], [2., 3.], [float("nan"), 1.], [6., 1.]], dtype=F64)
        weights = torch.tensor([1., 2**-45, -.5, 2., 1., .25, float("nan"), float("nan")], dtype=F64)
        es = [torch.tensor([0., 1., 2.], dtype=F64), torch.tensor([0., 1., 3.], dtype=F64)]
        data = xs.reshape(2,4,2).transpose(0,1)
        wt = weights.reshape(2,4).T
        with enable() as mode:
            gpu = xs.reshape(2,4,2).to("mps").transpose(0,1)
            gw = weights.reshape(2,4).to("mps").T
            for bins, options in [(es, {}), ([2, 3], {"range": [0.,2.,0.,3.]}),
                                  (2, {"range": [0.,2.,0.,3.]})]:
                arg = [e.to("mps") for e in es] if bins is es else bins
                for density in (False, True):
                    values, edges = torch.histogramdd(gpu, arg, weight=gw, density=density, **options)
                    ref = torch.histogramdd(data, bins, weight=wt, density=density, **options)
                    torch.testing.assert_close(values.cpu(), ref.hist, rtol=3e-15, atol=0, equal_nan=True)
                    for actual, expected in zip(edges, ref.bin_edges):
                        self.assertTrue(torch.equal(actual.cpu(), expected))
            hist = torch.histogramdd(gpu, [e.to("mps") for e in es], weight=gw).hist.cpu()
            self.assertEqual(hist[0,0].item(), .5 + 2**-45)
            self.assertFalse(mode.cpu_ops)

    def test_auto_empty_constant_and_out(self):
        with enable() as mode:
            for vals in [[], [4.25] * 11, [-3., -.5, 0., .2, 5.], [2**55] * 11]:
                host = torch.tensor(vals, dtype=F64)
                x = host.to("mps")
                self.assertTrue(torch.equal(torch.histc(x, 7).cpu(), torch.histc(host, 7)))
                if vals and vals[0] == 2**55:
                    # histc explicitly widens this constant interval by one ULP.
                    continue
                for density in (False, True):
                    ref = torch.histogram(host, 7, density=density)
                    dest = torch.empty(14, dtype=F64, device="mps")[::2]
                    bins = torch.empty(0, dtype=F64, device="mps")
                    got = torch.histogram(x, 7, density=density, out=(dest,bins))
                    self.assertIs(got.hist, dest)
                    self.assertIs(got.bin_edges, bins)
                    torch.testing.assert_close(dest.cpu(), ref.hist, equal_nan=True)
                    self.assertTrue(torch.equal(bins.cpu(), ref.bin_edges))
            empty = torch.empty(4, 0, dtype=F64, device="mps")
            self.assertEqual(torch.histogramdd(empty, []).hist.item(), 0)
            self.assertTrue(torch.histogramdd(empty, [], density=True).hist.isnan().item())
            nans = torch.tensor([1., float("nan")], dtype=F64, device="mps")
            for bad in (lambda: torch.histogram(nans, 3), lambda: torch.histc(nans, 3),
                        lambda: torch.histogram(nans, 0, range=(0.,1.)),
                        lambda: torch.histogram(nans, 3, range=(2.,1.)),
                        lambda: torch.histogram(nans, 3, range=(0.,float("inf"))),
                        lambda: torch.histogram(nans, torch.empty(1, dtype=F64, device="mps"))):
                with self.assertRaises(RuntimeError):
                    bad()
            self.assertFalse(mode.cpu_ops)

    def test_isin_scalars_sorted_small_and_out(self):
        x = torch.tensor([-0., 1., 1. + 2**-45, 3., float("nan"), float("inf")], dtype=F64)
        base = torch.tensor([0., 1. + 2**-45, float("nan"), float("inf")], dtype=F64)
        with enable() as mode:
            data = x.to("mps")
            for candidates in (base[:0], base, base.repeat(12)):
                tests = candidates.to("mps")
                for unique, invert in itertools.product((False,True), repeat=2):
                    # The large input intentionally repeats test elements, so
                    # compare to mathematical membership for both hints.
                    ref = torch.isin(x, candidates, assume_unique=False, invert=invert)
                    out = torch.empty(12, dtype=torch.bool, device="mps")[::2]
                    got = torch.isin(data, tests, assume_unique=unique, invert=invert, out=out)
                    self.assertIs(got,out)
                    self.assertTrue(torch.equal(got.cpu(),ref))
            self.assertTrue(torch.equal(torch.isin(data, 1.).cpu(), torch.isin(x, 1.)))
            self.assertTrue(torch.isin(0., base.to("mps")).item())
            for args in ((data, True), (True, data)):
                with self.assertRaises(RuntimeError):
                    torch.isin(*args)
            self.assertFalse(mode.cpu_ops)
