import matplotlib.pyplot as plt
import csv
f = open("/home/brabeem/Documents/deepLearning/DodgeBall_MA_RL/scripts/unityagents/rewards.txt",'r')
csv_reader = csv.reader(f)
agents = {"agent0":[],"agent1":[],"agent2":[],"agent3":[]}
for row in csv_reader:
    row = [float(el) for el in row]
    for i in range(4):
        agents["agent" + str(i)].append(row[i])
plt.subplot(221)
plt.plot(agents["agent" + str(0)])
plt.subplot(222)
plt.plot(agents["agent" + str(1)])
plt.subplot(223)
plt.plot(agents["agent" + str(2)])
plt.subplot(224)
plt.plot(agents["agent" + str(3)])
plt.show()