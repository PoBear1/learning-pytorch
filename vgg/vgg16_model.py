import	torch
import	torch.nn			as		nn
import	torch.optim			as		optim
import	torch.nn.functional	as		F
from	torch.utils.data	import	DataLoader

class vgg16_model(nn.Module):
	def __init__(self, dev: str = "cpu"):
		super().__init__()
		self.group1: nn.Sequential = nn.Sequential(
			nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size = 2, device = dev)
		)

		self.group2: nn.Sequential = nn.Sequential(
			nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size = 2, device = dev)
		)

		self.group3: nn.Sequential = nn.Sequential(
			nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, device = dev),	
			nn.ReLU(),
			nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, device = dev),	
			nn.ReLU(),
			nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, device = dev),	
			nn.ReLU(),
			nn.AvgPool2d(kernel_size = 2, device = dev)
		)

		self.group4: nn.Sequential = nn.Sequential(
			nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, device = dev),	
			nn.ReLU(),
			nn.AvgPool2d(kernel_size = 2, device = dev)
		)

		self.group5: nn.Sequential = nn.Sequential(
			nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, device = dev),
			nn.ReLU(),
			nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, device = dev),	
			nn.ReLU(),
			nn.AvgPool2d(kernel_size = 2, device = dev)
		)

		self.group6: nn.Sequential = nn.Sequential(
			nn.LazyLinear(out_features = 4096, bias = True, device = dev),
			nn.ReLU(),
			nn.Dropout(p = 0.5, inplace = False),	
			nn.Linear(in_features = 4096, out_features = 4096, bias = True, device = dev),
			nn.ReLU(),
			nn.Dropout(p = 0.5, inplace = False),	
			nn.Linear(in_features = 4096, out_features = 1000, bias = True, device = dev)
		)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		g1: torch.Tensor = self.group1(image)
		g2: torch.Tensor = self.group2(g1)
		g3: torch.Tensor = self.group3(g2)
		g4: torch.Tensor = self.group4(g3)
		g5: torch.Tensor = self.group5(g4)
		g5 = torch.flatten(g5, dim = 1)
		g6: torch.Tensor = self.group6(g5)
		return g6
class vgg16_trainer:
	def __init__(self, dataloader: tuple[DataLoader, DataLoader], lr: float = 0.001, momentum: float = 0.0, device: str = "cpu") -> None:
		self.model: nn.Module = (vgg16_model)(device).to(device)
		self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
		self.optimiser: optim.Optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = momentum)
		self.train_dataload: DataLoader = dataloader[0]
		self.test_dataload: DataLoader = dataloader[1]
		self.device: str = device
	def train(self) -> tuple[list[tuple[float, int]], int]:
		size: int = len(self.train_dataload.dataset)
		loss_history: list[tuple[float, int]] = []
		self.model.train()
		for batch, (X, y) in enumerate(self.train_dataload):
			X, y = X.to(self.device), y.to(self.device)
			latent: torch.Tensor = self.encode_model(X)
			pred: torch.Tensor = self.decode_model(latent)
			loss: torch.Tensor = self.loss_fn(pred, y)
			loss.backward()
			self.optimiser.step()
			self.optimiser.zero_grad()
			if (batch + 1) % 100 == 0:
				loss_history.append([loss.item(), (batch + 1) * len(X)])
		return [loss_history, size]
	def test(self) -> tuple[float, float]:
		size: int = len(self.test_dataload.dataset)
		num_batches: int = len(self.test_dataload)
		self.model.eval()
		test_loss: float = 0
		correct: int = 0
		with torch.no_grad():
			for X, y in self.test_dataload:
				X, y = X.to(self.device), y.to(self.device)
				latent: torch.Tensor = self.encode_model(X)
				pred: torch.Tensor = self.class_model(latent)
				test_loss += self.class_loss_fn(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()
		return [test_loss / num_batches, correct / size]
	def predict(self, image: torch.Tensor) -> torch.Tensor:
		self.model.eval()	
		with torch.no_grad():
			return self.class_model(self.encode_model(image))
	def drop_parameters(self, param_loc: str) -> None:
		torch.save(self.model.state_dict(), param_loc + "_vgg16.pth")