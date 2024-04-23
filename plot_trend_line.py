import numpy as np
import pickle
import matplotlib.pyplot as plt

dir_name='plots/2024-04-23_network7'
with open(dir_name +'/total_performence.pkl', 'rb') as file:
    total_performence = pickle.load(file)

with open(dir_name +'/iterations_arr.pkl', 'rb') as file:
    iterations_arr = pickle.load(file)

x = iterations_arr
y = total_performence

#create scatterplot
plt.subplots(figsize=(10, 5))
plt.scatter(x, y, color='blue')

#calculate equation for quadratic trendline
z = np.polyfit(x, y, 4)
p = np.poly1d(z)

#add trendline to plot
plt.plot(x, p(x), color='orange')

plt.xlabel('Iteration')
plt.ylabel('Mean score')
plt.savefig(dir_name +'/score_trend.png')