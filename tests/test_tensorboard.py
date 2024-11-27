import datetime

import numpy as np
from tensorboardX import SummaryWriter

time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fname = "/tmp/tb_test/" + time_str
writer = SummaryWriter(fname)


batch = 4
time = 64
channels = 3
height = 64
width = 64


for step in range(10):
    video = np.zeros((batch, time, channels, height, width))
    for t in range(time):
        for b in range(batch):
            for i in range(height):
                for j in range(width):
                    video[b, t, step % 3, i, j] = (
                        b / batch * np.sin(t / 10.0 + i / 10.0 + j / 10.0)
                        + np.random.randn() * 0.1
                    )

    writer.add_video("video", video, step, fps=10)
