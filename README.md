# Deep-Clustering-with-Convolutional-Autoencoders and a simple solution to the clustering problem( img2vec & cosine similarity)
Code from paper :  https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf


The implementation of this network was also taken as a basis : https://github.com/michaal94/torch_DCEC

# About CNN
This network uses the (Encoder x Decoder) with params ***[32,64,128]*** architecture, but there is a clustering layer between the layers.

<img src="https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-319-70096-0_39/MediaObjects/459886_1_En_39_Fig2_HTML.gif" width="400">

# Clustering Loss
<img src="https://deepnotes.io/public/images/AE-based.jpg" width="400">

# Train
**Encoder_Decoder** :
* Input : Tensor(batch_size, channels , (size,size)). **Default params : [16, 3, 128, 128]**
* Loss Function : torch.nn.MSE()
output = model(x_train)
loss = **Loss Function(output, x_train)**

# Other
This is just the first simple implementation of DCEC, improved implementations can be viewed from the links above.

* $ docker build -t $IMAGE_TAG -f ./Dockerfile ./

