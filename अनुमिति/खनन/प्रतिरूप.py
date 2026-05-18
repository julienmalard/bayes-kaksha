import pymc as pm
import pymc_extras as pmx
import arviz as az

from अनुमिति.खनन.आँकड़े import वर्ष, दूर्घटना_आँकड़ो

if __name__ == "__main__":
    with pm.Model() as disaster_model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=वर्ष.min(), upper=वर्ष.max())
        early_rate = pm.Exponential("early_rate", 1.0)
        late_rate = pm.Exponential("late_rate", 1.0)
        rate = pm.math.switch(switchpoint >= वर्ष, early_rate, late_rate)
        disasters = pm.Poisson("disasters", rate, observed=दूर्घटना_आँकड़ो)

    # disaster_model_marginalized = pmx.marginalize(disaster_model, ["switchpoint"])

    # with disaster_model_marginalized:
        परिणाम = pm.sample()
    #     परिणाम = pmx.recover_marginals(परिणाम)

    print(az.summary(परिणाम, var_names=["~disasters"], filter_vars="like"))

    # post = परिणाम.posterior.switchpoint.values.reshape(-1)
    az.plot_trace(परिणाम, var_names=["switchpoint", "early_rate", "late_rate"]).savefig("trace.png")
    az.plot_trace(परिणाम, var_names=["early_rate"]).savefig("trace early rate.png")
