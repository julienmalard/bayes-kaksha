import os.path

import arviz as az
import numpy as np
from pymc import HalfCauchy, Model, Normal, sample

from स्थिरांक import परिणाम_नत्थी


def आँकड़ों_बनाना():
    rng = np.random.default_rng()

    माप = 200
    true_intercept = 1
    true_slope = 2

    x = np.linspace(0, 1, माप)
    # y = a + b*x
    असली_रेखा = true_intercept + true_slope * x
    # add noise
    y = असली_रेखा + rng.normal(scale=0.5, size=माप)

    return [x, y]


if __name__ == '__main__':
    with Model():  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("Intercept", 0, sigma=20)
        slope = Normal("slope", 0, sigma=20)

        # Define likelihood
        x, y = आँकड़ों_बनाना()
        likelihood = Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        idata = sample(3000)

    fig = az.plot_trace(idata)
    fig.savefig(os.path.join(परिणाम_नत्थी, "जांच.png"))

print("सब ठीक ठाक")
