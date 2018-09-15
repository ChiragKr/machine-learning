# Apriori 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# apyori library requires input in a particular way:
# (1) Input is a list. Each element of this list is a transaction.
# (2) Each transaction is a list of items bought in that transaction.
# (3) Each item bought needs to be a string.
# From 1 and 2 we clearly see, input is "a list of lists".  

# Preparing dataset as required by the apyori library
transactions = []
for i in range(0,7501):
    transaction = [str(dataset.values[i, j]) for j in range(0,20)]
    transactions.append(transaction)
    
# Training the Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3 ,min_length=2)

# Visualizing the rules obtained by the apriori algorithm
results = list(rules)
# =============================================================================
print("Example:")
rule = str(set(results[0].ordered_statistics[0].items_base))
rule = rule + " ===> " + str(set(results[0].ordered_statistics[0].items_add))
print(f'RULE:\t' + rule +   
            '\nSUPPORT: %0.5f' % float(results[0][1]) +
            '\nCONF   : %0.5f' % float(results[0][2][0][2]) +
            '\nLIFT   : %0.5f' % float(results[0][2][0][3]))
print("SUPPORT = probabilty that a basket contains both 'chicken' and 'light cream'")
print("CONF    = probabilty that a person will buy 'chicken' given that they bought 'light cream'")
# =============================================================================
results_list = []
for i in range(0, len(results)):
    rule = str(set(results[i].ordered_statistics[0].items_base))
    rule = rule + " ===> " + str(set(results[i].ordered_statistics[0].items_add))
    results_list.append(f'RULE:\t' + rule +   
                        '\nSUPPORT: %0.5f' % float(results[i][1]) +
                        '\nCONF   : %0.5f' % float(results[i][2][0][2]) +
                        '\nLIFT   : %0.5f' % float(results[i][2][0][3]))