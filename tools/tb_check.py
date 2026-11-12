from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/tb/dummy")
for step in range(100):
    writer.add_scalar("dummy/loss", 1 / (step + 1), step)
writer.close()
