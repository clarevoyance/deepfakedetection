# CSE6242 Team 13: GAN Vector Database for Facial Similarity Search

## DESCRIPTION

This package runs a Streamlit app allowing the user to generate Images from a VAE
trained on CelebA and check their similarity against a prepopulated Vector Database
populated with CelebA. This is similar to a scenario, in which a fraudulent actor
tries to steal somebodies identity via a deep fake image.

The vector database and all similarity comparisons are made within the latent space of
an encoding neural network based on the Vision Transformer architecture.

A plot of this similarity space is shown at the bottom of the App allowing the user to
manually inspect similar and less similar images. Additionally, the basis created by
traversing the parameter space of the VAE - e. g. smile, age, etc. - are plotted as lines
within the similarity space giving the user further guidance to the semantics encoded.

## INSTALLATION

Start by downloading the necessary data files and pretrained networks and weights by running:
```
bash ./download.sh
```

We highly recommend, to use conda to create an environment with Python 3.7 and [install PyTorch](https://pytorch.org/get-started/locally/)
depending on the configuration of your system. For Apple Silicon (M1) users, Python 3.8 will work with the current CPU installation instructions.

The easiest way to install the dependencies is to run either `bash ./init-gpu.sh` or `bash ./init-cpu.sh`
You may also try the following in your terminal if you do not wish to run the bash scripts depending on whether you have a CUDA card installed. 

### For CUDA Installation
```
conda install -c huggingface transformers
conda install pandas numpy scikit-learn matplotlib opencv plotly bokeh==2.4.1
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install annoy streamlit
```

### For CPU Only Installation
```
conda install -c huggingface transformers
conda install pandas numpy scikit-learn matplotlib opencv plotly bokeh==2.4.1
pip install torch torchvision annoy streamlit
```

### Data
This project is built upon the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
The webservice expects the images of CelebA downloaded and placed in the data directory.

```
/data
├── img_align_celeba
│   ├── img_align_celeba
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── ...
├── identity_CelebA.txt
├── list_attr_celeba.csv
├── list_bbox_celeba.csv
├── list_eval_partition.csv
├── list_landmarks_align_celeba.csv
```

### Models
The Annoy Index file generated by the Index Builder notebook is to be expected in the root directory.

The InterFaceGAN models available in the [InterFaceGAN repository](https://github.com/genforce/interfacegan) 
are to be placed in the following directory.
`/interfacegan/models/pretrain/`

## EXECUTION
### Local Deployment
For your convenience, a `run.sh` bash script has been placed in the root folder. Running this will start the streamlit service.
Alternatively you can run the following command in your terminal in the root directory:
```
streamlit run ./app.py --server.enableCORS False
```
The first time you run it, streamlit may ask for your email. You can safely ignore it.

### GCP Deployment
A `Dockerfile` has been placed in the root directory if you wish to deploy the application onto your chosen cloud platform of choice.
In this example, we will show a simple to deploy this quickly on Google Cloud Platform's App Engine.

```
gcloud auth login
```
This will open the your browser to login with your relevant GCP credentials.
Once you have successfully logged in, you can create your project in your GCP interface.
```
gcloud config set project {PROJECT_NAME}
gcloud config set app/cloud_build_timeout 80000 
gcloud app deploy
```

You will be prompted to select the region of your virtual machine. This will take some time to deploy the App Engine. When you are done enter the following to open the web application in your browser:
```
gcloud app browse
```

## DEMO VIDEO
A quick demo of the local deployment and the web application's features can be viewed at https://www.youtube.com/watch?v=LptQBYvNtYg
