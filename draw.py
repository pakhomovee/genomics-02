import matplotlib.pyplot as plt
import numpy as np
f = open("runs/result.txt", 'r')
lines = f.readlines()
f.close()
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
                plt.scatter([N2] * len(arr), arr)
            Ns.append(N2)
            preds.append(np.average(arr))
    if mode == 'avg':
        plt.scatter(Ns, preds)
    plt.plot(Ns, Ns, color='r', alpha=0.5)
    plt.savefig(f'runs/{mode}.png')

draw(lines, 'avg')
draw(lines, 'all')
