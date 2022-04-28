import random
import numpy as np
from interfacegan.models.model_settings import MODEL_POOL
from interfacegan.models.pggan_generator import PGGANGenerator
from interfacegan.models.stylegan_generator import StyleGANGenerator
import torch


# !wget https://www.dropbox.com/s/nmo2g3u0qt7x70m/stylegan_celebahq.pth?dl=1 -O interfacegan/models/pretrain/stylegan_celebahq.pth --quiet

class VAE:
	def __init__(self):
		self.model_name = "stylegan_celebahq"
		self.latent_space_type = "W"
		self.generator = self.build_generator()
		self.ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
		self.boundaries = {}
		self.parameters = {
			"age": {"min": -3.0, "max": 3.0},
			"eyeglasses": {"min": -2.9, "max": 3.0},
			"gender": {"min": -3.0, "max": 3.0},
			"pose": {"min": -3.0, "max": 3.0},
			"smile": {"min": -3.0, "max": 3.0}
		}

		for i, attr_name in enumerate(self.ATTRS):
			boundary_name = f'{self.model_name}_{attr_name}'
			if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
				self.boundaries[attr_name] = np.load(f'interfacegan/boundaries/{boundary_name}_w_boundary.npy')
			else:
				self.boundaries[attr_name] = np.load(f'interfacegan/boundaries/{boundary_name}_boundary.npy')

	def build_generator(self):
		"""Builds the generator by model name."""
		gan_type = MODEL_POOL[self.model_name]['gan_type']
		if gan_type == 'pggan':
			generator = PGGANGenerator(self.model_name)
		elif gan_type == 'stylegan':
			generator = StyleGANGenerator(self.model_name)
		return generator

	def sample_codes(self, num, seed=0):
		"""Samples latent codes randomly."""
		np.random.seed(seed)
		codes = self.generator.easy_sample(num)
		if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
			codes = torch.from_numpy(codes).type(torch.FloatTensor).to(self.generator.run_device)
			codes = self.generator.get_value(self.generator.model.mapping(codes))
		return codes

	def generate_base_image(self, seed=None):
		if seed is None:
			seed = random.randint(0, 1000)

		latent_codes = self.sample_codes(1, seed)
		if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
			synthesis_kwargs = {'latent_space_type': 'W'}
		else:
			synthesis_kwargs = {}

		image = self.generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image'].squeeze()
		return image, latent_codes

	def generate_variational_image(self, latent_codes, parameter_dict):
		new_codes = latent_codes.copy()

		for parameter, value in parameter_dict.items():
			new_codes += self.boundaries[parameter] * value

		if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
			synthesis_kwargs = {'latent_space_type': 'W'}
		else:
			synthesis_kwargs = {}

		new_image = self.generator.easy_synthesize(new_codes, **synthesis_kwargs)['image'].squeeze()
		return new_image

	def generate_vae_axis(self, parameter_name, base_codes):
		assert parameter_name in self.parameters

		p_min = self.parameters[parameter_name]["min"]
		p_max = self.parameters[parameter_name]["max"]

		parameter_space = np.linspace(p_min, p_max, num=8)
		base_codes_copy = base_codes.copy().squeeze(axis=0)

		vae_batch_size = 4

		images = []
		for i in range(0, len(parameter_space), vae_batch_size):
			batch = parameter_space[i:i + vae_batch_size]  #

			new_codes = np.array([base_codes_copy + self.boundaries[parameter_name].squeeze(axis=0) * point
			                      for point in batch])

			if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
				synthesis_kwargs = {'latent_space_type': 'W'}
			else:
				synthesis_kwargs = {}

			# for image in self.generator.easy_synthesize(new_codes, **synthesis_kwargs)["image"]:
			# 	images.append(image)

			images.extend(self.generator.easy_synthesize(new_codes, **synthesis_kwargs)["image"].squeeze())

		return np.array(images)