import numpy as np
import pandas as pd
import os
import pymc as pm
from .उपकरण import scale

try:
    mhb_data = pd.read_csv(os.path.join("..", "data", "crossbill.csv"))
except FileNotFoundError:
    mhb_data = pd.read_csv(pm.get_data("crossbill.csv"))

year = 2001

# remove quadrats with missing surveys
is_y_column = mhb_data.columns.str.startswith(f"det{str(year)[2:]}")
y = mhb_data.loc[:, is_y_column].values
was_surveyed = ~np.isnan(y).any(axis=1)
y = y[was_surveyed].astype(int)
quadrat_count, survey_count = y.shape

# extract our covariates
elevation = mhb_data["ele"].values[was_surveyed]
forest_cover = mhb_data["forest"].values[was_surveyed]
is_date_column = mhb_data.columns.str.startswith(f"date{str(year)[2:]}")
date = mhb_data.loc[:, is_date_column].values[was_surveyed]

# prepare design matrix for the occurrence model
elev_scaled = scale(elevation)
forest_scaled = scale(forest_cover)
elev_scaled2 = elev_scaled ** 2
X = np.column_stack((np.ones_like(elevation), forest_scaled, elev_scaled, elev_scaled2))

# some dates are missing so we'll impute those
date[np.isnan(date)] = np.nanmedian(date)
W = np.stack((np.ones_like(date), date), axis=2)
fig, axes = plt.subplots(2, 2, figsize=(6, 5), tight_layout=True, sharey="row")
