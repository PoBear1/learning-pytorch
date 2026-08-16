import	torch
import	torch.nn 				as		nn
import	torch.optim				as		optim
import	torch.nn.functional		as		F
from	torch.utils.data		import	DataLoader

import	torchvision
from	torchvision.transforms	import	v2

import	vgg16_model				as		vgg16

from	sys						import	argv

batch_size: int = 100
auto_epochs: int = 100
train_epochs: int = 100
MNIST_data: tuple[DataLoader, DataLoader] = (
	torchvision.datasets.CIFAR10(
		root = "data", 
		train = True,
		download = True,
		transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
	),
	torchvision.datasets.CIFAR10(
		root = "data", 
		train = False,
		download = True,
		transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
	)
)

MNIST_dataloader: tuple[DataLoader, DataLoader] = (
	DataLoader(MNIST_data[0], batch_size = batch_size),
	DataLoader(MNIST_data[1], batch_size = batch_size)
)

auto_lr: float = 0.01

model: vgg16.vgg16_trainer = vgg16.vgg16_trainer(MNIST_dataloader, auto_lr, 0.0, device = "mps")
for t in range(auto_epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	history, size = model.train()
	for loss, current in history:
		print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
	test_loss = model.test()
	print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

model.drop_parameters("model_parameters")