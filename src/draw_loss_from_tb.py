from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt

log_dir = './log/noise_gan/lightning_logs/100/events.out.tfevents.1724205625.DESKTOP-2GBHHDM.7576.0'

event_acc = EventAccumulator(log_dir)
event_acc.Reload()

tags = event_acc.Tags()['scalars']
print(tags)

losses = {}
for tag in tags:
    event = event_acc.Scalars(tag)
    step = [e.step for e in event]
    value = [e.value for e in event]
    losses[tag] = (step, value)

for tag, (step, value) in losses.items():
    if tag == 'p_loss_epoch':
        plt.subplot(2, 3, 1)
        plt.plot(step, value, label=tag)
    if tag == 'd_loss_epoch':
        plt.subplot(2, 3, 2)
        plt.plot(step, value, label=tag)
    if tag == 'pp_loss_epoch':
        plt.subplot(2, 3, 3)
        plt.plot(step, value, label=tag)
    if tag == 'g_loss_epoch':
        plt.subplot(2, 3, 4)
        plt.plot(step, value, label=tag)
    if tag == 'G_Loss_epoch':
        plt.subplot(2, 3, 5)
        plt.plot(step, value, label=tag)
    if tag == 'D_Loss_epoch':
        plt.subplot(2, 3, 6)
        plt.plot(step, value, label=tag)

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图形
plt.show()
