import numpy as np
import matplotlib.pyplot as plt

file_name = './log/noise_gan/lightning_logs/100/epoch_loss.txt'

def plot(x, y, i, label):
    plt.subplot(2, 3, i)
    plt.plot(x, y, label=label)
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


i = 0
batch = []
Loss = {'p_loss': [], 'd_loss': [], 'pp_loss': [], 'g_loss': [], 'G_Loss': [], 'D_Loss': []}
with open(file_name, 'r') as f:
    line = f.readline()
    while line != '':
        print(i)
        str = line.split(':')
        if i % 7 == 0:
            batch.append(int(str[0]))
        else:
            Loss[str[0]].append(float(str[1]))
        i += 1
        line = f.readline()

print(len(batch), len(Loss['p_loss']))

plot(batch, Loss['p_loss'], 1, 'p_loss')
plot(batch, Loss['d_loss'], 2, 'd_loss')
plot(batch, Loss['pp_loss'], 3, 'pp_loss')
plot(batch, Loss['g_loss'], 4, 'g_loss')
plot(batch, Loss['G_Loss'], 5, 'G_Loss')
plot(batch, Loss['D_Loss'], 6, 'D_Loss')

# 显示图形
plt.show()
