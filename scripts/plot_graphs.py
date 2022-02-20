import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
purple_returns = []
blue_returns = []
purple_ratings = []
blue_ratings = []

with open('results.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
    
    for ROW in plotting:
        elo_rating = ast.literal_eval(ROW[2])
        blue_returns.append(float(ROW[0]))
        purple_returns.append(float(ROW[1]))
        purple_ratings.append(elo_rating[0])
        blue_ratings.append(elo_rating[1])

n_eps = len(purple_returns)
eps = range(n_eps)
 
plt.figure()
plt.plot(eps,blue_returns)
plt.plot(eps,purple_returns)
plt.xlabel('episodes')
plt.ylabel('avg returns')
plt.legend(['blue team return', 'purple team return'])
plt.savefig('returns.png')


plt.figure()
plt.plot(eps,blue_ratings)
plt.plot(eps,purple_ratings)
plt.xlabel('episodes')
plt.ylabel('elo ratings')
plt.legend(['blue team rating', 'purple team rating'])
plt.savefig('ratings.png')

      