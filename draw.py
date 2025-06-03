import matplotlib.pyplot as plt
import numpy as np
import json
import os
from params import GLOBAL_PARAMS
with open("data.json", "r") as f:
    data = json.load(f)
runId = data['runId']
data['runId'] = data.get('runId', 0) + 1
with open('data.json', 'w') as file:
    json.dump(data, file, indent=2)
os.mkdir(f'runs/{runId}')
f1 = open(f"runs/{runId}/result.txt", 'w')
f = open("runs/tmp.txt", 'r')
lines = f.readlines()
for line in lines:
    f1.write(line)
f1.close()
f.close()
with open(f"runs/{runId}/params.json", 'w') as f:
    json.dump(GLOBAL_PARAMS, f, indent=2)
def draw(lines, mode):
    Ns = []
    preds = []
    for line in lines:
        if len(line) >= 10:
            mem = line
            line = line.split()
            N2 = int(line[0].split('=')[1][:-1])
            line = mem
            arr = eval(line[line.find('['):line.find(']')+1])
            if mode == 'all':
                plt.scatter([N2] * len(arr), np.array(arr) * 2)
            Ns.append(N2)
            preds.append(np.average(arr))
            print(abs(N2 - preds[-1] * 2) / N2)
    if mode == 'avg':
        plt.scatter(Ns, np.array(preds) * 2)
    plt.plot(Ns, Ns, color='r', alpha=0.5)
    plt.savefig(f'runs/{runId}/{mode}.png')

draw(lines, 'avg')
draw(lines, 'all')
