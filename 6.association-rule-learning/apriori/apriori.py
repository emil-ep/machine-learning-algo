import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from apyori import apriori

dataset = pd.read_csv('6.association-rule-learning/apriori/Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#minimum support value (min_support)
#we have 7500 transactions, we need to find the strongest rule 
#min_support means ((the number of transactions by A and B)/total number of transactions)
# we want a transaction that happens 3 times a day, so for a whole week - 3*7 = 21 ; 21/7500 = 0.003
#For min_confidence start with 0.8 or reduce it accordingly to get some value. 
#min_lift - 3
#For out business case, we need to develop the outcome as 2 (buy this get that free model)
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, 
min_length = 2, max_length = 2)

results = list(rules)
# print(results)
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# print(resultsinDataFrame)

#sorting the results by the decreasing value in Lift column
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))