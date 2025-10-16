import torch
import torch.autograd as autograd
from .components import Generator, Critic
from .base_gan import BaseGAN

torch.set_float32_matmul_precision("medium")


class WGAN_GP(BaseGAN):
    def __init__(
        self,
        latent_dim=100,
        g_every_k_steps=1,
        lambda_gp=10.0,
        distribution_name=None,
        distribution_params=None,
        num_test_samples=1000,
    ):
        super().__init__(
            latent_dim=latent_dim,
            g_every_k_steps=g_every_k_steps,
            distribution_name=distribution_name,
            distribution_params=distribution_params,
            num_test_samples=num_test_samples,
        )
        self.generator = Generator(latent_dim)
        self.discriminator = Critic()
        self.lambda_gp = lambda_gp
        # TODO: make these hyperparams configurable
        self.lr = 0.0002
        self.betas = (0.5, 0.999)

        self.save_hyperparameters()
        self.apply(self._init_weights)

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)

        # create alpha with the correct number of singleton dims so expand/broadcast works
        alpha_shape = (batch_size,) + (1,) * (real_samples.dim() - 1)
        alpha = torch.rand(alpha_shape, device=self.device)

        # detach fake_samples so GP does not flow back to the generator
        fake_samples_det = fake_samples.detach()

        interpolates = (
            alpha * real_samples + (1.0 - alpha) * fake_samples_det
        ).requires_grad_(True)
        interp_scores = self.discriminator(interpolates)

        # grad_outputs must match interp_scores shape
        grad_outputs = torch.ones_like(interp_scores, device=self.device)
        gradients = autograd.grad(
            outputs=interp_scores,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)  # reshape is safer than view
        grad_norm = gradients.norm(2, dim=1)
        gp = self.lambda_gp * ((grad_norm - 1.0) ** 2).mean()
        return gp

    def compute_discriminator_loss(self, real_samples, fake_samples):
        # basic WGAN loss
        real_scores = self.discriminator(real_samples)
        fake_scores = self.discriminator(fake_samples)
        loss = fake_scores.mean() - real_scores.mean()

        gp = self.compute_gradient_penalty(real_samples, fake_samples)
        return loss + gp

    def compute_generator_loss(self, fake_samples):
        fake_scores = self.discriminator(fake_samples)
        return -fake_scores.mean()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return opt_g, opt_d
