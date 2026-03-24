## Problem and goal
We want to learn a **conditional distribution generator** for the [MFTR](https://arxiv.org/pdf/2212.02448) distribution, that has parameters (ranges are approximate based on physical interpretation):

$$
\begin{align*}
    m & \in [1, 8] \\
    \mu & \in [1, 8] \\
    K & \in [0, 20] \\
    \Delta & \in [0, 1] \\
    \bar{\gamma} & \in [1, 2]
\end{align*}
$$ 

Thus, our final goal is to generate samples that match the target envelope distribution closely.

<div class="callout-info">
    <strong>Milestones</strong>
    <ul>
        <li> Current: learn this on synthetic MFTR data. </li>
        <li> Long-term goal: transfer the same pipeline to unknown real-world channel measurements, then interpolate/extrapolate across parameter settings. </li>
    </ul>
</div>

## What Has Been Done
- Conditional GAN with MLP generator/discriminator and learned condition embeddings.
- Dataset built by **uniformly sampling parameter combinations** over configured ranges.
- Current focused experiment: vary $\mu$ in training range $[1,9]$ (others fixed).
- Training setup: BCE-with-logits, label smoothing, Adam, and more frequent D updates than G.
- Evaluation: QQ + density/PDF overlays and metrics (MAE, MSE, KS, CvM, Wasserstein).

## Main References
1. [GAN connoisseur: Can GANs learn simple 1D parametric distributions](https://chunliangli.github.io/docs/dltp17gan.pdf)
1. [Time series simulation by conditional generative adversarial net](https://arxiv.org/abs/1904.11419)
1. [The Multi-Cluster Fluctuating Two-Ray Fading Model](https://arxiv.org/pdf/2212.02448)
1. [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

## Tried So Far
- WGAN and WGAN-GP. They yield the same results as [1]: mode localization is preserved when unimodal, but all density concentrates there and the region nearby. Not appropiate for this problem.
- Using the hinge loss instead of BCE produced worse results. 

## Next Steps
- Extend training to full multi-parameter ranges (all 5 MFTR parameters varied).
- Spectral normalization in discriminator (and compare with other normalization variants such as batch norm).
- Use projection discriminator instead of direct concat.
- Try MMDA-style GAN objectives, as in [1].
- Add a parameter to BaseCGAN to choose the generator and discriminator architectures. This will aid in comparison as well.
- Add a parameter to the CGAN class to control the loss used, so that we can easily compare how each one performs.
- Save all training and configuration details for later reference.