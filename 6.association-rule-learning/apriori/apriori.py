import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('6.association-rule-learning/apriori/Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

