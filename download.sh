#!/bin/bash
# downloads the large files that cannot be stored on Github

echo "====== DOWNLOAD IN PROGRESS ======"
echo "This may take a while..."
echo ""
echo "STATUS: Downloading CelebA dataset..."
wget https://storage.googleapis.com/cse6242-team13/setup.sh/celeba.zip
echo "STATUS: CelebA dataset downloaded!"
echo "STATUS: Unzipping CelebA dataset... This may take some time."
unzip -qq -n celeba.zip
rm celeba.zip

echo "STATUS: Dataset unzipped!"
echo "STATUS: Downloading pretrained BEiT and vision transformer network..."
wget https://storage.googleapis.com/cse6242-team13/setup.sh/celeba_beit_index.ann
wget https://storage.googleapis.com/cse6242-team13/setup.sh/celeba_vit_finetuned_index.ann
echo "STATUS: Transformer networks downloaded!"

cd interfacegan/models
mkdir pretrain
cd pretrain
echo "STATUS: Downloading pretrained weights for StyleGAN..."
wget https://storage.googleapis.com/cse6242-team13/setup.sh/stylegan_celebahq.pth
echo "STATUS: Pretrained weights downloaded!"
echo "Thank you for your patience!"
echo "====== DOWNLOAD COMPLETED ======"
echo "You may now proceed to install the dependencies."