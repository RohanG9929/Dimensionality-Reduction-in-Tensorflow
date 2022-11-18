# Dimensionality Reduction in Tensorflow
 
Developing an Autoencoder to extract crucial features within its hidden layers.


Below is the original image followed by the reconstruction performed by the decoder from the encoder extracted features.


<img src="https://user-images.githubusercontent.com/103215628/202778348-0a1f7e72-9d77-475f-8f92-58ac50834450.png" width="640" height="256">

![LowDim]()





Below is the reduction of the dimensionality of the dataset down to 2. Now the data can be visualised on a 2D plane. The encoder was able to extarct enough features to be able to roughly classify the images into distinct categories. These categories are shown with the colours below. 

<img src="https://user-images.githubusercontent.com/103215628/202778355-36b4a5ae-ac69-46a6-9a49-050d7b842e0c.png" width="640" height="640">


The same 2-Dimensional representation, with each image shown in its respective class. It is clear to see how the encoder classified each type of clothing.!

<img src="https://user-images.githubusercontent.com/103215628/202778363-f951bb6c-84d1-4f18-89bd-e06b8caad484.png" width="640" height="640">
