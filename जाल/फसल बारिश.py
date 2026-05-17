from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Step 1: Define the network structure.
सिंचाई_प्रतिरूप = DiscreteBayesianNetwork(
    [
        ("सिंचाई", "फसल गीली"),
        ("बारिश", "फसल गीली"),
    ]
)

# Step 2: Define the CPDs.
cpd_बारिश = TabularCPD(
    variable="बारिश", variable_card=2, values=[[0.2], [0.8]],
    state_names={
        "बारिश": ["हुआ", "नहीं हुआ"],
    }
)
print("===== बारिश =====")
print(cpd_बारिश)

cpd_सिंचाई = TabularCPD(
    variable="सिंचाई", variable_card=2, values=[[0.1, 0.3], [0.9, 0.7]],
    evidence=["बारिश"],
    evidence_card=[2],
    state_names={
        "सिंचाई": ["किया", "नहीं किया"],
        "बारिश": ["हुआ", "नहीं हुआ"],
    }
)
print("===== सिंचाई =====")
print(cpd_सिंचाई)

# Step 3: Add the CPDs to the model.
सिंचाई_प्रतिरूप.add_cpds(cpd_बारिश, cpd_सिंचाई)

# Step 4: Check if the model is correctly defined.
सिंचाई_प्रतिरूप.check_model()

अनुमिति = VariableElimination(सिंचाई_प्रतिरूप)
inferred_distribution = अनुमिति.query(variables=['बारिश'], evidence={'फसल गीली': "गीली"})
print(inferred_distribution)
