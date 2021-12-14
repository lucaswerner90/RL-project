import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize((84,84), interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	screen = env.render(mode='rgb_array').transpose((2, 0, 1))
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	screen = resize(screen).unsqueeze(0).to(device)
	return screen

