import base64

from bokeh.plotting import ColumnDataSource, figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid


class BokehPlots:
	def plot_2d_v2(self, center_image_vector, center_image,
	               neighbor_vectors: list, neighbor_images, neighbor_identities,
	               random_vectors: list, random_images, random_identities,
	               center_identity=None, dimensionality_reduction="pca", vae_lines=None):

		point_vectors = neighbor_vectors + [center_image_vector] + random_vectors
		pca = self.reduction_v2(point_vectors, technique=dimensionality_reduction)
		point_results = pca.transform(point_vectors)

		line_vectors = []
		if vae_lines:
			for vae_vectors in vae_lines.values():
				line_vectors += list(vae_vectors)
			line_results = pca.transform(line_vectors)
		else:
			line_results = line_vectors

		vae_lines_reduced = {}
		vae_chunk_start = 0
		if vae_lines:
			for parameter, vae_vectors in vae_lines.items():
				vae_lines_reduced[parameter] = line_results[vae_chunk_start:vae_chunk_start + len(vae_vectors)]
				vae_chunk_start += len(vae_vectors)

		encoded_images = []
		for image in neighbor_images + [Image.fromarray(center_image)] + random_images:
			byte_io = BytesIO()
			image.save(byte_io, format='png')
			encoded_image = byte_io.getvalue()
			url = 'data:image/png;base64,' + base64.b64encode(encoded_image).decode('utf-8')
			encoded_images.append(url)

		identities = neighbor_identities + ["Generated Image"] + random_identities

		colors = ["blue"] * len(neighbor_vectors) + ["red"] + ["green"] * len(random_vectors)
		legends = ["neighbor"] * len(neighbor_vectors) + ["generated"] + ["random"] * len(random_vectors)
		if dimensionality_reduction == 'pca':
			title = '2D PCA Interactive Plot with Images'
			source = ColumnDataSource(data=dict(
				x=point_results.T[0],
				y=point_results.T[1],
				desc=identities,
				imgs=encoded_images,
				color=colors,
				legend=legends
			))
		else:
			title = '2D TSNE Interactive Plot with Images'
			source = ColumnDataSource(data=dict(
				x=point_results.T[0],
				y=point_results.T[1],
				desc=identities,
				imgs=encoded_images,
				color=colors,
				legend=legends
			))

		TOOLTIPS = """
                <div>
                    <div>
                        <img
                            src="@imgs" height="128" width="128"
                            style="float: left; margin: 0px 15px 15px 0px;"
                            border="2"
                        ></img>
                    </div>
                    <div>
                        <span style="font-size: 17px;">Identity: @desc</span>
                    </div>
                </div>
                        """

		p = figure(width=900, height=450, tooltips=TOOLTIPS, title=title, align='center')

		colors = ["teal", "pink", "brown", "orange", "purple"]
		for i, (parameter, vae_line) in enumerate(vae_lines_reduced.items()):
			p.line(vae_line.T[0], vae_line.T[1], line_width=2, color=colors[i], legend_label=parameter)

		p.circle('x', 'y', size=15, source=source, color="color", legend_group="legend")

		return p

	def reduction_v2(self, vectors, random_state=6242, technique='pca', n_components=2):
		if technique == 'pca':
			points = PCA(n_components=n_components, random_state=random_state).fit(vectors)
		else:
			points = TSNE(n_components=n_components, verbose=0, random_state=random_state).fit(vectors)

		return points

	def plot_neighbors(self, neighbor_images, neighbor_identities):
		fig = plt.figure(figsize=(8, 3), dpi=140)
		grid = ImageGrid(fig, 111, nrows_ncols=(1, len(neighbor_identities)), axes_pad=0.05)

		# Draw results
		for ax, image, identity in zip(grid, neighbor_images, neighbor_identities):
			ax.imshow(np.array(image))
			ax.axis('off')
			ax.set_title(str(identity))

		return fig
