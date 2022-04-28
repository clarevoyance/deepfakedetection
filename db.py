from annoy import AnnoyIndex
import pandas as pd
from collections import defaultdict
from PIL import Image


class VectorDB:
	def __init__(self, index_path="./celeba_beit_index.ann"):
		self.DIM = 768
		self.index = AnnoyIndex(self.DIM, 'euclidean')
		self.index.load(index_path)

		# TODO: Hardcoded path
		identities = pd.read_csv("./data/identity_CelebA.txt", sep=" ", header=None)
		identities.rename(columns={0: "file", 1: "identity"}, inplace=True)
		identity_selection = identities.identity.unique()

		df = identities[identities.identity.isin(identity_selection)].reset_index()
		idx_to_identity = df.to_dict('index')
		identity_to_idx = defaultdict(list)

		image_paths, keys = [], []
		for k, v in idx_to_identity.items():
			identity_to_idx[v["identity"]].append(k)
			image_paths.append("data/img_align_celeba/img_align_celeba/" + v["file"])
			keys.append(k)

		self.idx_to_identity = idx_to_identity
		self.identity_to_idx = identity_to_idx

	def lookup_by_vector(self, vector, num_results=10, include_distances=False):
		assert len(vector) == self.DIM

		return self.index.get_nns_by_vector(vector, num_results, search_k=-1, include_distances=include_distances)

	def lookup_by_idx(self, idx, num_results=10, include_distances=False):

		return self.index.get_nns_by_item(idx, num_results, search_k=-1, include_distances=include_distances)

	def load_image(self, idx):
		return Image.open("data/img_align_celeba/img_align_celeba/" + self.idx_to_identity[idx]["file"])

