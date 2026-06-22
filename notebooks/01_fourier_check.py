"""
A sanity check for the analysis method itself, not the model. Run:

    python notebooks/01_fourier_check.py

The claim in analyze.py is that a "spiky" Fourier spectrum means something real
was learned. Before trusting that on the trained model, check that the measure
behaves: a random embedding should give a roughly FLAT spectrum (no frequency
special), while a matrix built by hand out of two cosine waves should light up
at exactly those two frequencies. If this check passes, a spiky spectrum on the
real model is genuine structure and not an artefact of the DFT.
"""
import numpy as np

p, d = 113, 128
rng = np.random.default_rng(0)


def spectrum(mat):
    s = np.linalg.norm(np.abs(np.fft.fft(mat, axis=0)), axis=1)[: p // 2 + 1]
    s[0] = 0.0
    return s


# 1) random matrix -> flat-ish, no single frequency dominates
rand = rng.standard_normal((p, d))
sr = spectrum(rand)
peak_ratio = sr.max() / sr.mean()
print(f"random matrix: peak/mean = {peak_ratio:.2f}  (near 1 means flat, as expected)")
assert peak_ratio < 3.0

# 2) matrix planted with waves at k=7 and k=41 -> those two should dominate
n = np.arange(p)[:, None]
planted = np.cos(2 * np.pi * 7 * n / p) + np.cos(2 * np.pi * 41 * n / p)
planted = planted * rng.standard_normal((1, d))     # give each dim its own phase/scale
sp = spectrum(planted)
top2 = sorted(int(k) for k in np.argsort(sp)[::-1][:2])
print(f"planted waves at [7, 41], recovered top-2 frequencies: {top2}")
assert top2 == [7, 41]
print("fourier method behaves, spikes on the real model mean real structure")
