# https://www.pymc.io/projects/examples/en/stable/case_studies/occupancy.html
import pymc_extras as pmx

import os
import arviz as az

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.ticker import PercentFormatter
from matplotlib import colors

rng = np.random.default_rng()

ax = axes.flat[0]
ax.hist(elevation, ec="w", bins=7)
ax.set_ylabel("Number of quadrats")
ax.set_title("Elevation (m)")

ax = axes.flat[1]
ax.hist(forest_cover, ec="w", bins=7, fc="C2")
ax.set_title("Forest cover")
ax.xaxis.set_major_formatter(PercentFormatter())

ax = axes.flat[2]
real_date = np.datetime64("2001-01-01") + (date - 1).astype("timedelta64[D]")
df = pd.DataFrame({"date": real_date.flat})
monthly_counts = df.groupby(df["date"].dt.to_period("M")).size()
ax.bar(monthly_counts.index.strftime("%b"), monthly_counts.values, ec="w", fc="C1")
ax.set_ylabel("Number of surveys")
ax.set_title("Date")

axes.flat[3].remove()

plt.show()

coords = {
    "survey_effects": ["intercept", "date"],
    "quadrat_effects": ["intercept", "forest_cover", "elevation", "elevation2"],
    "quadrats": np.arange(quadrat_count),
    "surveys": np.arange(survey_count),
}

with pm.Model(coords=coords) as occupancy:

    # occurrence probability model
    beta = pm.Normal("beta", 0, 2, dims="quadrat_effects")
    occurrence_probability = pm.math.invlogit(pm.math.dot(X, beta))

    # detection probability model
    alpha = pm.Normal("alpha", 0, 2, dims="survey_effects")
    detection_probability = pm.math.invlogit(pm.math.dot(W, alpha))

    # occupied / unoccupied state at each site
    z = pm.Bernoulli("z", occurrence_probability, dims="quadrats")

    # likelihood
    pm.Bernoulli("y", z[:, None] * detection_probability, dims=["quadrats", "surveys"], observed=y)

pm.model_to_graphviz(occupancy)

# marginalize the model before sampling
occupancy_marginal = pmx.marginalize(occupancy, ["z"])
with occupancy_marginal:
    occupancy_idata = pm.sample()

az.summary(occupancy_idata)

az.plot_rank_dist(occupancy_idata)

# get the posterior predictive distribution for the in sample data
with occupancy_marginal:
    predictions = pm.sample_posterior_predictive(
        occupancy_idata, predictions=True
    ).predictions

# create scaled values for predictions and corresponding real values for plotting
scaled_elev_plot = np.linspace(min(elev_scaled), max(elev_scaled), 100)
elev_plot = scaled_elev_plot * elevation.std() + elevation.mean()
scaled_forest_plot = np.linspace(min(forest_scaled), max(forest_scaled), 100)
forest_cover_plot = scaled_forest_plot * forest_cover.std() + forest_cover.mean()

# predict the occurrence probability conditional on average values for other predictors
beta = az.extract(occupancy_idata, var_names="beta").values
elev_pred = invlogit(
    beta[0][:, None] + beta[2][:, None] * scaled_elev_plot + beta[3][:, None] * scaled_elev_plot**2
)
forest_pred = invlogit(beta[0][:, None] + beta[1][:, None] * scaled_forest_plot)

# predict the detection probability over the course of the year
alpha = az.extract(occupancy_idata)["alpha"].values
plot_date = np.arange(date.min(), date.max())
p_pred = invlogit(alpha[0][:, None] + alpha[1][:, None] * plot_date)

# convert the day of year to actual dates
base_date = np.datetime64("2001-01-01")
dates = base_date + (plot_date - 1).astype("timedelta64[D]")

fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey=True, layout="constrained")

# plot 1000 samples from the posterior
sample_indices = rng.choice(len(alpha[0]), size=500)
for sample in sample_indices:
    axes[0, 0].plot(elev_plot, elev_pred[sample], alpha=0.05, color="C0")
    axes[0, 1].plot(forest_cover_plot, forest_pred[sample], alpha=0.05, color="C2")
    axes[1, 0].plot(dates, p_pred[sample], color="C1", alpha=0.05)

axes[0, 0].set_ylim((0, 1))

axes[0, 0].set_ylabel(r"Occurrence probability", fontsize=13)
axes[1, 0].set_ylabel(r"Detection probability", fontsize=13)

axes[1, 0].set_title(r"Day of year")
axes[0, 0].set_title("Elevation (m)")
axes[0, 1].set_title("Forest cover")

axes[1, 0].set_xticks(["2001-01-15", "2001-02-15", "2001-03-15", "2001-04-15"])
axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[0, 1].xaxis.set_major_formatter(PercentFormatter())

axes[1, 1].remove()

plt.show()

try:
    suisse = pd.read_csv(os.path.join("..", "data", "switzerland.csv"))
except FileNotFoundError:
    suisse = pd.read_csv(pm.get_data("switzerland.csv"))

suisse.head()

# scale the maps elevation using the mhb statistics
map_elev_scaled = (suisse.elevation.values - elevation.mean()) / elevation.std()
map_forest_scaled = (suisse.forest.values - forest_cover.mean()) / forest_cover.std()

# predict the occurrence probability across the country
psi_tilde_map = invlogit(
    beta[0][:, None]
    + beta[1][:, None] * map_forest_scaled
    + beta[2][:, None] * map_elev_scaled
    + beta[3][:, None] * map_elev_scaled**2
)
eti = az.eti(psi_tilde_map.T, prob=0.9)

suisse["psi_hat"] = psi_tilde_map.mean(axis=0)
suisse["psi_low"] = eti[:, 0]
suisse["psi_high"] = eti[:, 1]
suisse["psi_se"] = np.std(psi_tilde_map, axis=0)

fig, ax = plt.subplots(figsize=(7, 6))

scat = ax.scatter(suisse.x, suisse.y, marker="s", s=1, c=suisse.psi_hat, cmap="viridis")
ax.set_aspect("equal")
ax.set_facecolor("w")
ax.set_ylabel("Northing (m)")
ax.set_xlabel("Easting (m)")
ax.set_title("Distribution of Red Crossbills in Switzerland")
fig.colorbar(scat, ax=ax, label=r"Occurrence probability $\psi$", fraction=0.03)
plt.ticklabel_format(axis="both", style="sci", scilimits=(4, 4))
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)

norm = colors.Normalize(vmin=np.min(eti), vmax=np.max(eti))

ax = axes[0]
scat = ax.scatter(
    suisse.x, suisse.y, marker="s", s=0.5, c=suisse.psi_low, cmap="viridis", norm=norm
)
ax.set_aspect("equal")
ax.set_facecolor("w")
ax.set_ylabel("Northing (m)")
ax.set_title("Lower (5%)")

ax = axes[1]
scat = ax.scatter(
    suisse.x, suisse.y, marker="s", s=0.5, c=suisse.psi_high, cmap="viridis", norm=norm
)
ax.set_aspect("equal")
ax.set_facecolor("w")
ax.set_ylabel("Northing (m)")
ax.set_xlabel("Easting (m)")
ax.set_title("Upper (95%)")
plt.ticklabel_format(axis="both", style="sci", scilimits=(4, 4))

fig.colorbar(
    scat, ax=axes, orientation="vertical", fraction=0.03, label=r"Occurrence probability $\psi$"
)

fig, ax = plt.subplots(figsize=(5, 5))

scat = ax.scatter(suisse.x, suisse.y, marker="s", s=1, c=suisse.psi_se, cmap="magma")
ax.set_aspect("equal")
ax.set_facecolor("w")
ax.set_ylabel("Northing (m)")
ax.set_xlabel("Easting (m)")
ax.set_title("Standard error")
fig.colorbar(scat, ax=ax, fraction=0.03)
plt.ticklabel_format(axis="both", style="sci", scilimits=(4, 4))
plt.show()
