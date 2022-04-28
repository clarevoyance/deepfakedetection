import torch
from torch import nn
import torchvision
from torchvision import transforms
from transformers import ViTForImageClassification
from transformers import AutoModelForImageClassification
from transformers import ViTFeatureExtractor, BeitFeatureExtractor

models = [
	'facebook/deit-base-distilled-patch16-224',
	'microsoft/beit-base-patch16-224-pt22k-ft22k',
	'google/vit-base-patch16-224',
	# 'openai/imagegpt-small', # ->  super slow
	# 'openai/imagegpt-large', # -> out of RAM error
	# 'facebook/detr-resnet-50-panoptic', # -> not able to find timm package
	# 'facebook/vit-mae-base' -> strange error during index.add_item()
]


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class LatentSpace:
	def __init__(self, model_identifier='microsoft/beit-base-patch16-224-pt22k-ft22k'):
		assert model_identifier in models
		self.model_identifier = model_identifier
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.encoder = AutoModelForImageClassification.from_pretrained(self.model_identifier)
		self.encoder.classifier = Identity()
		self.encoder.eval()
		self.encoder.to(self.device)

		# TODO: Change to AutoFeaureExtractor
		self.feature_extractor = BeitFeatureExtractor.from_pretrained(self.model_identifier)

	def transform(self, image):
		embedding = self.transform_batch(image).squeeze()

		return embedding

	def transform_batch(self, images):
		encoding = self.feature_extractor(images=images, return_tensors="pt")
		pixel_values = encoding['pixel_values'].to(self.device)
		outputs = self.encoder(pixel_values)
		embedding = outputs.logits.detach().cpu().numpy()

		return embedding
