import math
from pomegranate import *


intelligence = DiscreteDistribution({'Thap': 0.7, 'Cao': 0.3})
difficulty = DiscreteDistribution({'De': 0.6, 'Kho': 0.4})
grade = ConditionalProbabilityTable(
    [['Thap', 'De', 'A', 0.3],
     ['Thap', 'De', 'B', 0.4],
     ['Thap', 'De', 'C', 0.3],
     ['Thap', 'Kho', 'A', 0.05],
     ['Thap', 'Kho', 'B', 0.25],
     ['Thap', 'Kho', 'C', 0.7],
     ['Cao', 'De', 'A', 0.9],
     ['Cao', 'De', 'B', 0.08],
     ['Cao', 'De', 'C', 0.02],
     ['Cao', 'Kho', 'A', 0.5],
     ['Cao', 'Kho', 'B', 0.3],
     ['Cao', 'Kho', 'C', 0.2]],
    [intelligence, difficulty])

sat = ConditionalProbabilityTable(
    [['Thap', 'Thap', 0.95],
     ['Thap', 'Cao', 0.05],
     ['Cao', 'Thap', 0.2],
     ['Cao', 'Cao', 0.8]],
    [intelligence])

letter = ConditionalProbabilityTable(
    [['A', 'Yeu', 0.1],
     ['A', 'Manh', 0.9],
     ['B', 'Yeu', 0.4],
     ['B', 'Manh', 0.6],
     ['C', 'Yeu', 0.99],
     ['C', 'Manh', 0.01]],
    [grade])

d1 = State(intelligence, name="intelligence")
d2 = State(difficulty, name="difficulty")
d3 = State(grade, name="grade")
d4 = State(sat, name="sat")
d5 = State(letter, name="letter")

# Building the Bayesian Network
network = BayesianNetwork("validation")
network.add_states(d1, d2, d3, d4, d5)
network.add_edge(d1, d3)
network.add_edge(d1, d4)
network.add_edge(d3, d5)
network.add_edge(d2, d3)
network.bake()

beliefs = network.predict_proba({'grade': 'A'})
print(beliefs)
