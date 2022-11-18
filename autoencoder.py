
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


tf.random.set_seed(42)
np.random.seed(42)

########################################################
#Loading the Fashion MNIST Data
########################################################
fashion_mnist = tf.keras.datasets.fashion_mnist

#Training set labels are not imported into a variable here as the learning is unsupervised.
(images, _), (images_test, test_labels) = fashion_mnist.load_data()
images = images.astype(np.float32) / 255

#The 10000 test images will be used to visualise he output
images_test = images_test.astype(np.float32) / 255

#First 50000 for training and last 10000 for validation
images_train, images_valid = images[:int(10*images.shape[0]/12)], images[int(10*images.shape[0]/12):]


########################################################
#Creating the Encoder and Decoder models for Unsupervised Training
########################################################

class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten(input_shape=[28, 28])
        self.linear1 = tf.keras.layers.Dense(128, activation='selu')
        self.linear2 = tf.keras.layers.Dense(30, activation='selu')
        
    def call(self, x):
        return self.linear2(self.linear1(self.flatten(x)))

class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.linear1 = tf.keras.layers.Dense(100, activation='selu',input_shape=[30])
        self.linear2 = tf.keras.layers.Dense(28*28, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape([28,28])
        
    def call(self, x):
        return self.reshape(self.linear2(self.linear1(x)))


myEncoder = Encoder()
myDecoder = Decoder()


#Combining to create an Autoencoder
myTrainingModel = tf.keras.models.Sequential([myEncoder, myDecoder])

########################################################
#Compiling and Training the Autoencoder
########################################################
myTrainingModel.compile(loss=tf.keras.losses.MeanSquaredError(),
                   optimizer=keras.optimizers.SGD(learning_rate=1.5), metrics='mean_squared_error')

myTrainingModel.fit(images_train, images_train, epochs=5, validation_data=(images_valid, images_valid))


#Showing the low dimensional output of my model
idx = np.random.randint(images_test.shape[0], size=10)
plt.figure(figsize=(10, 4))
lowDim = myTrainingModel.predict(images[idx])
for i in range(1,10,1):
    plt.subplot(2, 10, i);plt.imshow(images[idx[i-1]])
    plt.gray()
    plt.axis("off")
    plt.subplot(2, 10, i+10);plt.imshow(lowDim[i-1])
    plt.gray()
    plt.axis("off")

plt.savefig('LowDim.png')
plt.show()
########################################################
#Visualising all 10,000 images in one single visualisation
########################################################
X_rep = myEncoder.predict(images_test)

tsne = TSNE()
X_viz_in_2D = tsne.fit_transform(X_rep)

#Normalising between 0 and 1
X_viz_in_2D = (X_viz_in_2D - X_viz_in_2D.min()) / (X_viz_in_2D.max() - X_viz_in_2D.min())

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.scatter(X_viz_in_2D[:, 0], X_viz_in_2D[:, 1], c=test_labels, s=10, cmap='tab10')
plt.savefig('allTestIms_scatter_viz.png')
plt.show()

plt.figure(figsize=(10, 10))
for idx, position in enumerate(X_viz_in_2D):
    plt.gca().add_artist(
        matplotlib.offsetbox.AnnotationBbox(
        matplotlib.offsetbox.OffsetImage(images_test[idx], cmap="binary"),
        position, bboxprops={"edgecolor": plt.cm.tab10(test_labels[idx])})
    )
    

plt.axis("off")
plt.savefig('allTestIms_viz.png')
plt.show()

print("Done")