import	torch
import	torch.nn			as		nn
import	torch.optim			as		optim
import	torch.nn.functional	as		F
from	torch.utils.data	import	DataLoader

class mlp_encoder(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.layer1: nn.Linear = nn.Linear(in_features = 28 * 28, out_features = 100, bias = True, device = device)
		self.layer2: nn.Linear = nn.Linear(in_features = 100, out_features = 50, bias = True, device = device)
		self.layer3: nn.Linear = nn.Linear(in_features = 50, out_features = 10, bias = True, device = device)
		self.layer4: nn.Linear = nn.Linear(in_features = 10, out_features = 2, bias = True, device = device)
	def forward(self, input_image: torch.Tensor) -> torch.Tensor:
		in_image: torch.Tensor = torch.flatten(input_image, start_dim = 1)
		layer1_output: torch.Tensor = nn.ReLU()(self.layer1(in_image))
		layer2_output: torch.Tensor = nn.ReLU()(self.layer2(layer1_output))
		layer3_output: torch.Tensor = nn.ReLU()(self.layer3(layer2_output))
		return self.layer4(layer3_output)

class mlp_decoder(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.layer1: nn.Linear = nn.Linear(in_features = 2, out_features = 10, bias = True, device = device)
		self.layer2: nn.Linear = nn.Linear(in_features = 10, out_features = 50, bias = True, device = device)
		self.layer3: nn.Linear = nn.Linear(in_features = 50, out_features = 100, bias = True, device = device)
		self.layer4: nn.Linear = nn.Linear(in_features = 100, out_features = 28 * 28, bias = True, device = device)
	def forward(self, input_image: torch.Tensor) -> torch.Tensor:
		# in_image: torch.Tensor = torch.flatten(input_image, start_dim = 1)
		layer1_output: torch.Tensor = nn.ReLU()(self.layer1(input_image))
		layer2_output: torch.Tensor = nn.ReLU()(self.layer2(layer1_output))
		layer3_output: torch.Tensor = nn.ReLU()(self.layer3(layer2_output))
		return torch.reshape(nn.ReLU()(self.layer4(layer3_output)), (-1, 1, 28, 28))

class mlp_classifier(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.layer1: nn.Linear = nn.Linear(in_features = 2, out_features = 3, bias = True, device = device)
		self.layer2: nn.Linear = nn.Linear(in_features = 3, out_features = 5, bias = True, device = device)
		self.layer3: nn.Linear = nn.Linear(in_features = 5, out_features = 7, bias = True, device = device)
		self.layer4: nn.Linear = nn.Linear(in_features = 7, out_features = 10, bias = True, device = device)
	def forward(self, latents: torch.Tensor) -> torch.Tensor:
		layer1_output: torch.Tensor = nn.ReLU()(self.layer1(latents))
		layer2_output: torch.Tensor = nn.ReLU()(self.layer2(layer1_output))
		layer3_output: torch.Tensor = nn.ReLU()(self.layer3(layer2_output))
		return nn.ReLU()(self.layer4(layer3_output))

class cnn_encoder(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.cnn_layer1: nn.Conv2d = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 7, stride = 1, padding = 0)

class cnn_arch(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.layer1_features: nn.Conv2d = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 7, stride = 1, device = device)
		self.layer2_features: nn.Conv2d = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 5, stride = 1, device = device)
		self.layer3_features: nn.Conv2d = nn.Conv2d(in_channels = 20, out_channels = 64, kernel_size = 3, stride = 1, device = device)
		self.mlp_layer1: nn.LazyLinear = nn.LazyLinear(out_features = 50, bias = True, device = device)
		self.mlp_layer2: nn.Linear = nn.Linear(in_features = 50, out_features = 20, bias = True, device = device)
		self.mlp_layer3: nn.Linear = nn.Linear(in_features = 20, out_features = 10, bias = True, device = device)
	def forward(self, input_image: torch.Tensor) -> torch.Tensor:
		featurised_1: torch.Tensor = self.layer1_features(input_image)
		featurised_2: torch.Tensor = self.layer2_features(featurised_1)
		featurised_3: torch.Tensor = self.layer3_features(featurised_2)
		featurised_3 = torch.flatten(featurised_3, start_dim = 1)
		layer1_inference: torch.Tensor = nn.ReLU()(self.mlp_layer1(featurised_3))
		layer2_inference: torch.Tensor = nn.ReLU()(self.mlp_layer2(layer1_inference))
		layer3_inference: torch.Tensor = nn.ReLU()(self.mlp_layer3(layer2_inference))
		return layer3_inference

class mlp_autoencoder_trainer:
	def __init__(self, dataloader: tuple[DataLoader, DataLoader], auto_lr: float = 0.001, class_lr: float = 0.001, auto_momentum: float = 0.0, class_momentum: float = 0.0, device: str = "cpu") -> None:
		self.encode_model: nn.Module = (mlp_encoder)(device).to(device)
		self.decode_model: nn.Module = (mlp_decoder)(device).to(device)
		self.class_model: nn.Module = (mlp_classifier)(device).to(device)
		self.encode_loss_fn: nn.CrossEntropyLoss = nn.MSELoss(reduction = 'sum')
		self.class_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
		self.auto_optimiser: optim.Optimizer = optim.SGD(
			list(self.encode_model.parameters()) + list(self.decode_model.parameters()), 
			lr = auto_lr, momentum = auto_momentum
		)
		self.class_optimiser: optim.Optimizer = optim.SGD(
			self.class_model.parameters(), 
			lr = class_lr, momentum = class_momentum	
		)
		self.train_dataload: DataLoader = dataloader[0]
		self.test_dataload: DataLoader = dataloader[1]
		self.device: str = device
	def train_autoencoder(self) -> tuple[list[tuple[float, int]], int]:
		size: int = len(self.train_dataload.dataset)
		loss_history: list[tuple[float, int]] = []
		self.encode_model.train()
		self.decode_model.train()
		for batch, (X, _) in enumerate(self.train_dataload):
			X = X.to(self.device)
			latent: torch.Tensor = self.encode_model(X)
			pred: torch.Tensor = self.decode_model(latent)
			loss: torch.Tensor = self.encode_loss_fn(pred, X)
			loss.backward()
			self.auto_optimiser.step()
			self.auto_optimiser.zero_grad()
			if (batch + 1) % 100 == 0:
				loss_history.append([loss.item(), (batch + 1) * len(X)])
		return [loss_history, size]
	def train_classifier(self) -> tuple[list[tuple[float, int]], int]:
		size: int = len(self.train_dataload.dataset)
		loss_history: list[tuple[float, int]] = []
		for p in self.encode_model.parameters():
			p.requires_grad = False
		self.encode_model.eval()
		self.class_model.train()
		for batch, (X, y) in enumerate(self.train_dataload):
			X, y = X.to(self.device), y.to(self.device)
			with torch.no_grad():
				latent: torch.Tensor = self.encode_model(X)
			pred: torch.Tensor = self.class_model(latent)
			loss: torch.Tensor = self.class_loss_fn(pred, y)
			loss.backward()
			self.class_optimiser.step()
			self.class_optimiser.zero_grad()
			if (batch + 1) % 100 == 0:
				loss_history.append([loss.item(), (batch + 1) * len(X)])
		return [loss_history, size]
	def test_autoencoder(self) -> float:
		num_batches: int = len(self.test_dataload)
		self.encode_model.eval()
		self.decode_model.eval()
		self.class_model.eval()
		test_loss: float = 0
		with torch.no_grad():
			for X, _ in self.test_dataload:
				X = X.to(self.device)
				latent: torch.Tensor = self.encode_model(X)
				pred: torch.Tensor = self.decode_model(latent)
				test_loss += self.encode_loss_fn(pred, X).item()
		return test_loss / num_batches
	def test_classifier(self) -> tuple[float, float]:
		size: int = len(self.test_dataload.dataset)
		num_batches: int = len(self.test_dataload)
		self.encode_model.eval()
		self.decode_model.eval()
		self.class_model.eval()
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
		self.encode_model.eval()
		self.class_model.eval()	
		with torch.no_grad():
			return self.class_model(self.encode_model(image))
	def generate(self, latent: torch.Tensor) -> torch.Tensor:
		self.decode_model.eval()
		with torch.no_grad():
			return self.decode_model(latent)
	def drop_parameters(self, param_loc: str) -> None:
		torch.save(self.encode_model.state_dict(), param_loc + "_encode.pth")
		torch.save(self.decode_model.state_dict(), param_loc + "_decode.pth")
		torch.save(self.class_model.state_dict(), param_loc + "_class.pth")