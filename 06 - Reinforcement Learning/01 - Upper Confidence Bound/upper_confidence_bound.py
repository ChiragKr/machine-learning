# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Actually we don't start with any data, the dataset
# above data is like a simulation collected at the end.
# Each row is a round. user will click on the ad version if 1 otherwise not

# Implementing UCB
import math
N = 10000
d = 10

ads_selected = [] # ad selected at each round
number_of_selections = [0] * d # N_i(n) : number of times ad i was selected up to n rounds
sums_of_rewards = [0] * d # R_i(n) : the sum of rewards of the ad i up to round n.
total_reward = 0

for n in range(N): # looping over round 'n'
    ad = 0 # keeps track of which ad is the best at each round
    max_upper_bound = 0
    for i in range(d): # looping over as 'i'
        if (number_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if max_upper_bound < upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] += reward
    total_reward += reward 

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()