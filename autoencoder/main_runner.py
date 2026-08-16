import	torch
import	torch.nn 				as		nn
import	torch.optim				as		optim
import	torch.nn.functional		as		F
from	torch.utils.data		import	DataLoader

import	torchvision
from	torchvision.datasets	import	VisionDataset
from	torchvision.transforms	import	v2

from	auto_model				import	mlp_encoder, mlp_decoder, mlp_classifier

import	numpy					as		np
import	pandas					as		pd
import	matplotlib.pyplot		as		plt
import	seaborn					as		sns
import	plotly.express			as		px

print("Setting up model...")

encode_model: mlp_encoder = mlp_encoder("mps")
decode_model: mlp_decoder = mlp_decoder("mps")
class_model: mlp_classifier = mlp_classifier("mps")

print("Initialising model...")

encode_model.load_state_dict(torch.load('model_parameters_encode.pth', weights_only = True))
decode_model.load_state_dict(torch.load('model_parameters_decode.pth', weights_only = True))
class_model.load_state_dict(torch.load('model_parameters_class.pth', weights_only = True))

dev: str = "mps"

print(f"Moving model onto {dev}...")

encode_model.to(device = dev)
decode_model.to(device = dev)
class_model.to(device = dev)

print("Setting model to eval...")

encode_model.eval()
decode_model.eval()
class_model.eval()

latent_version: list[tuple[torch.Tensor, int, np.ndarray]] = []

print("Setting datasets up...")

MNIST_data: tuple[VisionDataset, VisionDataset] = (
	torchvision.datasets.MNIST(
		root = "data", 
		train = True,
		download = True,
		transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
	),
	torchvision.datasets.MNIST(
		root = "data", 
		train = False,
		download = True,
		transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
	)
)

print("Setting dataloaders up...")

MNIST_dataloader: tuple[DataLoader, DataLoader] = (
	DataLoader(MNIST_data[0], batch_size = 1),
	DataLoader(MNIST_data[1], batch_size = 1)
)

def generate_latent(image: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		image = image.to(dev)
		return encode_model(image)
	
def generate_image(latent: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		latent = latent.to(dev)
		return decode_model(latent)

def generate_class(latent: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		latent = latent.to(dev)
		return class_model(latent)

def classify(image: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		image = image.to(dev)
		return class_model(encode_model(image))
	
if __name__ == "__main__":
	for X, y in MNIST_dataloader[1]:
		latents: np.ndarray = generate_latent(X).to('cpu').numpy()	
		latent_version.append(([float(latents[:, 0][0]), float(latents[:, 1][0])], int(y[0]), X.squeeze().to('cpu').numpy()))

	latent = [x[0] for x in latent_version]
	number = [str(x[1]) for x in latent_version]
	image = [x[2] for x in latent_version]

	frame = pd.DataFrame({
		'latent_0' : [x[0] for x in latent],
		'latent_1' : [x[1] for x in latent],
		'number': number,
		'image': image
	})
	fig = px.scatter(frame, x = 'latent_0', y = 'latent_1', color = 'number')
	fig.show()

	fig = px.imshow(image[0])
	fig.show()

	n = 20
	digit_size = 28
	grid_image = np.zeros((digit_size * n, digit_size * n))

	grid_x = np.linspace(-1, 1, n)   # plain linear spacing — no ppf
	grid_y = np.linspace(-1, 1, n)

	with torch.no_grad():
		for i, yi in enumerate(grid_y):
			for j, xi in enumerate(grid_x):
				z = torch.tensor([[xi, yi]], dtype = torch.float32, device = dev)
				x_decoded = decode_model(z)
				digit = x_decoded.squeeze().cpu().numpy()

				grid_image[
					i * digit_size : (i + 1) * digit_size,
					j * digit_size : (j + 1) * digit_size
				] = digit

	fig = px.imshow(grid_image, origin='upper')
	fig.update_xaxes(visible = False)
	fig.update_yaxes(visible = False)
	fig.show()