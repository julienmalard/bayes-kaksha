# Source: https://github.com/pgmpy/pgmpy_tutorials/blob/master/notebooks/11.%20A%20Bayesian%20Network%20to%20model%20the%20influence%20of%20energy%20consumption%20on%20greenhouse%20gases%20in%20Italy.ipynb?short_path=962f808
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

model = DiscreteBayesianNetwork([('Pop', 'EC'), ('Urb', 'EC'), ('GDP', 'EC'),
                                 ('EC', 'FFEC'), ('EC', 'REC'), ('EC', 'EI'),
                                 ('REC', 'CO2'), ('REC', 'CH4'), ('REC', 'N2O'),
                                 ('FFEC', 'CO2'), ('FFEC', 'CH4'), ('FFEC', 'N2O')])

from pandas import read_csv, DataFrame
import numpy as np


def annual_growth(row, years):
    min_year = years["min"]
    max_year = years["max"]
    row["Indicator Name"] = row["Indicator Name"] + " - [annual growth %]"
    for year in range(max_year, min_year, -1):
        if not np.isnan(row[str(year)]) and not np.isnan(row[str(year - 1)]):
            row[str(year)] = 100 * (float(row[str(year)]) - float(row[str(year - 1)])) / abs(float(row[str(year - 1)]))
        else:
            row[str(year)] = np.nan
    row[str(min_year)] = np.nan
    return row


years = {"min": 1960, "max": 2019}
df_raw = read_csv("./csv/italy-raw-data.csv")
df_raw_growth = DataFrame(
    data=[row if "growth" in row["Indicator Name"] else annual_growth(row, years) for index, row in df_raw.iterrows()])
print("There are " + str(df_raw_growth.shape[0]) + " indicators in the dataframe.")
df_raw_growth.head()

nodes = ['Pop', 'Urb', 'GDP', 'EC', 'FFEC', 'REC', 'EI', 'CO2', 'CH4', 'N2O']
df_growth = df_raw_growth.transpose().iloc[4:]
df_growth.columns = nodes
df_growth.head(10)

TIERS_NUM = 3


def boundary_str(start, end, tier):
    return f'{tier}: {start:+0,.2f} to {end:+0,.2f}'


def relabel(v, boundaries):
    if v >= boundaries[0][0] and v <= boundaries[0][1]:
        return boundary_str(boundaries[0][0], boundaries[0][1], tier='A')
    elif v >= boundaries[1][0] and v <= boundaries[1][1]:
        return boundary_str(boundaries[1][0], boundaries[1][1], tier='B')
    elif v >= boundaries[2][0] and v <= boundaries[2][1]:
        return boundary_str(boundaries[2][0], boundaries[2][1], tier='C')
    else:
        return np.nan


def get_boundaries(tiers):
    prev_tier = tiers[0]
    boundaries = [(prev_tier[0], prev_tier[prev_tier.shape[0] - 1])]
    for index, tier in enumerate(tiers):
        if index != 0:
            boundaries.append((prev_tier[prev_tier.shape[0] - 1], tier[tier.shape[0] - 1]))
            prev_tier = tier
    return boundaries


new_columns = {}
for i, content in enumerate(df_growth.items()):
    (label, series) = content
    values = np.sort(np.array([x for x in series.tolist() if not np.isnan(x)], dtype=float))
    if values.shape[0] < TIERS_NUM:
        print(f'Error: there are not enough data for label {label}')
        break
    boundaries = get_boundaries(tiers=np.array_split(values, TIERS_NUM))
    new_columns[label] = [relabel(value, boundaries) for value in series.tolist()]

df = DataFrame(data=new_columns)
df.columns = nodes
df.index = range(years["min"], years["max"] + 1)
df.head(10)

# disable text wrapping in output cell

model.cpds = []
estimator = DiscreteBayesianEstimator(
    prior_type="BDeu",
    equivalent_sample_size=10,
)

model.fit(data=df, estimator=estimator)

print(f'Check model: {model.check_model()}\n')
for cpd in model.get_cpds():
    print(f'CPT of {cpd.variable}:')
    print(cpd, '\n')

# Network analysis

print(f'There can be made {len(model.get_independencies().get_assertions())}',
      'valid independence assertions with respect to the all possible given evidence.')
print(
    'For instance, any node in the network is independent of its non-descendents given its parents (local semantics):\n',
    f'\n{model.local_independencies(nodes)}\n')


def active_trails_of(query, evidence):
    active = model.active_trail_nodes(query, observed=evidence).get(query)
    active.remove(query)
    if active:
        if evidence:
            print(f'Active trails between \'{query}\' and {active} given the evidence {set(evidence)}.')
        else:
            print(f'Active trails between \'{query}\' and {active} given no evidence.')
    else:
        print(f'No active trails for \'{query}\' given the evidence {set(evidence)}.')


def markov_blanket_of(node):
    print(f'Markov blanket of \'{node}\' is {set(model.get_markov_blanket(node))}')


active_trails_of(query='FFEC', evidence=[])
active_trails_of(query='FFEC', evidence=['EC'])
active_trails_of(query='FFEC', evidence=['EC', 'CO2'])
active_trails_of(query='EI', evidence=['EC'])
active_trails_of(query='CH4', evidence=['EC', 'FFEC'])
print()
markov_blanket_of(node='Pop')
markov_blanket_of(node='EC')
markov_blanket_of(node='REC')
markov_blanket_of(node='CH4')


def independent_assertions_score_function(node):
    return len([a for a in model.get_independencies().get_assertions() if node in a.event1])


def evidence_assertions_score_function(node):
    return len([a for a in model.get_independencies().get_assertions() if node in a.event3])


def update(assertion_dict, node, score_function):
    tmp_score = score_function(node)
    if tmp_score == assertion_dict["max"]["score"]:
        assertion_dict["max"]["nodes"].append(node)
    elif tmp_score > assertion_dict["max"]["score"]:
        assertion_dict["max"]["nodes"] = [node]
        assertion_dict["max"]["score"] = tmp_score
    if tmp_score == assertion_dict["min"]["score"]:
        assertion_dict["min"]["nodes"].append(node)
    elif tmp_score < assertion_dict["min"]["score"]:
        assertion_dict["min"]["nodes"] = [node]
        assertion_dict["min"]["score"] = tmp_score


if len(nodes) > 1:
    independent_init = independent_assertions_score_function(nodes[0])
    independent_dict = {"max": {"nodes": [nodes[0]], "score": independent_init},
                        "min": {"nodes": [nodes[0]], "score": independent_init}}
    evidence_init = evidence_assertions_score_function(nodes[0])
    evidence_dict = {"max": {"nodes": [nodes[0]], "score": evidence_init},
                     "min": {"nodes": [nodes[0]], "score": evidence_init}}
    for node in nodes[1:]:
        update(independent_dict, node, independent_assertions_score_function)
        update(evidence_dict, node, evidence_assertions_score_function)

print(f'Nodes which appear most ({independent_dict["max"]["score"]} times) in independence assertions',
      f'as independent variable are:\n{set(independent_dict["max"]["nodes"])}')
print(f'Nodes which appear least ({independent_dict["min"]["score"]} times) in independence assertions',
      f'as independent variable are:\n{set(independent_dict["min"]["nodes"])}')
print(f'Nodes which appear most ({evidence_dict["max"]["score"]} times) in independence assertions',
      f'as evidence are:\n{set(evidence_dict["max"]["nodes"])}')
print(f'Nodes which appear least ({evidence_dict["min"]["score"]} times) in independence assertions',
      f'as evidence are:\n{set(evidence_dict["min"]["nodes"])}')

from pgmpy.independencies.Independencies import IndependenceAssertion

def check_assertion(model, independent, from_variables, evidence):
    assertion = IndependenceAssertion(independent, from_variables, evidence)
    result = False
    for a in model.get_independencies().get_assertions():
        if frozenset(assertion.event1) == a.event1 and assertion.event2 <= a.event2 and frozenset(assertion.event3) == a.event3:
            result = True
            break
    print(f'{assertion}: {result}')

check_assertion(model, independent=['EI'], from_variables=['N2O'], evidence=['EC'])
check_assertion(model, independent=['EI'], from_variables=["FFEC", "REC", "GDP", "Pop", "Urb"], evidence=['EC'])
check_assertion(model, independent=['EC'], from_variables=["CH4"], evidence=['FFEC'])
check_assertion(model, independent=['EC'], from_variables=["CH4"], evidence=['FFEC', 'REC'])
check_assertion(model, independent=['FFEC'], from_variables=["REC"], evidence=['EC'])
check_assertion(model, independent=['FFEC'], from_variables=["REC"], evidence=['EC', 'CO2'])
check_assertion(model, independent=['CH4'], from_variables=["CO2"], evidence=['FFEC'])
check_assertion(model, independent=['CH4'], from_variables=["CO2"], evidence=['FFEC', 'REC'])

from pgmpy.inference import VariableElimination
import time


def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    if desc:
        print(desc)
    start_time = time.time()
    print(infer.query(variables=variables,
                      evidence=evidence,
                      elimination_order=elimination_order,
                      show_progress=show_progress))
    print(f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')


def get_ordering(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    start_time = time.time()
    ordering = infer._get_elimination_order(variables=variables,
                                            evidence=evidence,
                                            elimination_order=elimination_order,
                                            show_progress=show_progress)
    if desc:
        print(desc, ordering, sep='\n')
        print(f'--- Ordering found in {time.time() - start_time:0,.4f} seconds ---\n')
    return ordering


def padding(heuristic):
    return (heuristic + ":").ljust(16)


def compare_all_ordering(infer, variables, evidence=None, show_progress=False):
    ord_dict = {
        "MinFill": get_ordering(infer, variables, evidence, "MinFill", show_progress),
        "MinNeighbors": get_ordering(infer, variables, evidence, "MinNeighbors", show_progress),
        "MinWeight": get_ordering(infer, variables, evidence, "MinWeight", show_progress),
        "WeightedMinFill": get_ordering(infer, variables, evidence, "WeightedMinFill", show_progress)
    }
    if not evidence:
        pre = f'elimination order found for probability query of {variables} with no evidence:'
    else:
        pre = f'elimination order found for probability query of {variables} with evidence {evidence}:'
    if ord_dict["MinFill"] == ord_dict["MinNeighbors"] and ord_dict["MinFill"] == ord_dict["MinWeight"] and ord_dict[
        "MinFill"] == ord_dict["WeightedMinFill"]:
        print(f'All heuristics find the same {pre}.\n{ord_dict["MinFill"]}\n')
    else:
        print(f'Different {pre}')
        for heuristic, order in ord_dict.items():
            print(f'{padding(heuristic)} {order}')
        print()


infer = VariableElimination(model)

var = ['GDP']
heuristic = "MinNeighbors"
ordering = get_ordering(infer, variables=var, elimination_order=heuristic,
                        desc=f'Elimination order for {var} with no evidence computed through {heuristic} heuristic:')
query_report(infer, variables=var, elimination_order=ordering,
             desc=f'Probability query of {var} with no evidence through precomputed elimination order:')
query_report(infer, variables=var, elimination_order=list(reversed(ordering)),
             desc=f'Probability query of {var} with no evidence through dummy elimination order:')
compare_all_ordering(infer, variables=var)

var = ['CO2']
ev = {'EC': 'A: -7.05 to -0.12'}
heuristic = "MinFill"
query_report(infer, variables=var, evidence=ev, elimination_order=heuristic,
             desc=f'Probability query of {var} with evidence {ev} computed through {heuristic} heuristic:')
compare_all_ordering(infer, variables=var, evidence=ev)
heuristic = "MinNeighbors"
query_report(infer, variables=var, evidence=ev, elimination_order=heuristic,
             desc=f'Probability query of {var} with evidence {ev} computed through {heuristic} heuristic:')