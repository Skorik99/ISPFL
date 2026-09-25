from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import torch


class ExactMaternGP(gpytorch.models.ExactGP):
    """Exact GP with the Matérn-5/2 kernel used by Algorithm 8."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, inputs):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(inputs), self.covar_module(inputs)
        )


@dataclass(frozen=True)
class GPPrediction:
    mean: np.ndarray
    standard_deviation: np.ndarray


class GPSurrogate:
    def __init__(
        self,
        training_steps: int,
        restarts: int,
        learning_rate: float,
        jitter: float,
        seed: int,
    ):
        self.training_steps = training_steps
        self.restarts = restarts
        self.learning_rate = learning_rate
        self.jitter = jitter
        self.seed = seed
        self.model = None
        self.likelihood = None

    def fit(self, x, y, noise_variance) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        noise_variance = np.asarray(noise_variance, dtype=np.float64)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-12] = 1.0
        self.y_mean = float(y.mean())
        self.y_scale = max(float(y.std()), 1e-8)

        train_x = torch.as_tensor((x - self.x_mean) / self.x_scale, dtype=torch.float64)
        train_y = torch.as_tensor((y - self.y_mean) / self.y_scale, dtype=torch.float64)
        train_noise = torch.as_tensor(
            np.maximum(noise_variance / self.y_scale**2, 1e-8),
            dtype=torch.float64,
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            best_loss = float("inf")
            best_state = None
            for restart in range(self.restarts):
                torch.manual_seed(self.seed + restart)
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=train_noise, learn_additional_noise=True
                ).double()
                model = ExactMaternGP(train_x, train_y, likelihood).double()
                if restart:
                    with torch.no_grad():
                        model.mean_module.constant.normal_(0.0, 0.5)
                        model.covar_module.raw_outputscale.normal_(0.0, 0.5)
                        model.covar_module.base_kernel.raw_lengthscale.normal_(0.0, 0.5)
                loss = self._optimize(model, likelihood, train_x, train_y)
                if loss < best_loss:
                    best_loss = loss
                    best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(likelihood.state_dict()))
        finally:
            torch.set_num_threads(previous_threads)

        self.train_x = train_x
        self.train_y = train_y
        self.train_noise = train_noise
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=train_noise, learn_additional_noise=True
        ).double()
        self.model = ExactMaternGP(train_x, train_y, self.likelihood).double()
        self.model.load_state_dict(best_state[0])
        self.likelihood.load_state_dict(best_state[1])
        return best_loss

    def _optimize(self, model, likelihood, train_x, train_y) -> float:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        with gpytorch.settings.cholesky_jitter(self.jitter):
            for _ in range(self.training_steps):
                optimizer.zero_grad()
                loss = -objective(model(train_x), train_y)
                loss.backward()
                optimizer.step()
        return float(loss.detach())

    def predict(self, x) -> GPPrediction:
        test_x = torch.as_tensor(
            (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale,
            dtype=torch.float64,
        )
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(self.jitter):
            posterior = self.model(test_x)
        return GPPrediction(
            mean=posterior.mean.cpu().numpy() * self.y_scale + self.y_mean,
            standard_deviation=posterior.stddev.cpu().numpy() * self.y_scale,
        )

    def condition(self, x, y, noise_variance) -> None:
        self.predict(x)
        inputs = torch.as_tensor(
            (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale,
            dtype=torch.float64,
        )
        targets = torch.as_tensor(
            (np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_scale,
            dtype=torch.float64,
        )
        noise = torch.as_tensor(
            np.maximum(np.asarray(noise_variance) / self.y_scale**2, 1e-8),
            dtype=torch.float64,
        )
        self.model = self.model.get_fantasy_model(inputs, targets, noise=noise)
        self.likelihood = self.model.likelihood

    def clone(self) -> "GPSurrogate":
        clone = GPSurrogate(
            self.training_steps,
            self.restarts,
            self.learning_rate,
            self.jitter,
            self.seed,
        )
        clone.x_mean = self.x_mean.copy()
        clone.x_scale = self.x_scale.copy()
        clone.y_mean = self.y_mean
        clone.y_scale = self.y_scale
        clone.train_x = self.model.train_inputs[0].detach().clone()
        clone.train_y = self.model.train_targets.detach().clone()
        clone.train_noise = self.model.likelihood.noise_covar.noise.detach().clone()
        clone.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=clone.train_noise, learn_additional_noise=True
        ).double()
        clone.model = ExactMaternGP(clone.train_x, clone.train_y, clone.likelihood).double()
        clone.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        clone.likelihood.load_state_dict(copy.deepcopy(self.likelihood.state_dict()))
        return clone

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
                "train_x": self.model.train_inputs[0],
                "train_y": self.model.train_targets,
                "train_noise": self.model.likelihood.noise_covar.noise,
                "x_mean": self.x_mean,
                "x_scale": self.x_scale,
                "y_mean": self.y_mean,
                "y_scale": self.y_scale,
            },
            path,
        )
