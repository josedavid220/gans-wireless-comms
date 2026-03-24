from __future__ import annotations


def overview_markdown() -> str:
    return r"""
## Problem and Goal
We want to learn a **conditional distribution generator**: given MFTR parameters
$[m,\mu,K,\delta,\omega]$, generate samples that match the target envelope distribution.

Current milestone: learn this on synthetic MFTR data.
Long-term goal: transfer the same pipeline to unknown real-world channel measurements,
then interpolate/extrapolate across parameter settings.

## What Has Been Done (Concise)
- Conditional GAN with MLP generator/discriminator and learned condition embeddings.
- Dataset built by **uniformly sampling parameter combinations** over configured ranges.
- Current focused experiment: vary **$\mu$** in training range $[1,9]$ (others fixed).
- Training setup: BCE-with-logits, label smoothing, Adam, and more frequent D updates than G.
- Evaluation: QQ + density/PDF overlays and metrics (MAE, MSE, KS, CvM, Wasserstein).

## Main References
- **Primary references (placeholders):** `fu2019time`, `zaheer2017gan`
- MFTR model reference: `sánchez2023multiclusterfluctuatingtworayfading`
- Foundational GAN/cGAN: `goodfellow2014generativeadversarialnetworks`, `mirza2014conditional`

## Tried So Far (Preliminary)
- WGAN / WGAN-GP / hinge variants explored with limited gains.
- Observed issue (consistent with prior 1D GAN discussions): harder shape coverage around transitions and tails.

## Next Steps
- Spectral normalization in discriminator (and compare with other normalization variants).
- Projection discriminator instead of direct concat.
- Try MMDA-style GAN objectives.
- Extend training to full multi-parameter ranges (all 5 MFTR parameters varied).
"""


def tried_methods_markdown() -> str:
    return r"""
### Tried Methods Snapshot (Placeholder Cards)
- **BCE cGAN (current baseline):** best overall behavior so far.
- **WGAN / WGAN-GP:** limited improvement in this 1D setting.
- **Hinge loss:** tested, no clear robustness gain yet.

(Comparison panels can be attached here once curated assets are exported.)
"""


def metrics_help_markdown() -> str:
    return r"""
### Metrics (v1)
- **MAE / MSE (quantiles):** pointwise discrepancy between sorted generated vs real samples.
- **KS statistic / p-value:** maximum ECDF gap and its significance.
- **CvM statistic / p-value:** integrated ECDF mismatch and significance.
- **Wasserstein distance:** 1D earth mover distance.

(AD is intentionally excluded in v1 for MFTR.)
"""
