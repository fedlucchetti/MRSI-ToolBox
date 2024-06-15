import tensorflow as tf
from tensorflow.keras import layers, models


class DeepDenoiser(object):
    def __init__(self) -> None:
        pass

    def build_autoencoder(self,input_shape):
        # Encoder
        net = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv3D(64, kernel_size = (3,3,3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2), padding='same'),
            layers.Conv3D(32, kernel_size = (3,3,3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2), padding='same'),

 
            layers.Conv3DTranspose(32, kernel_size = (3,3,3), strides=2,  activation='relu', padding='same'),
            layers.Conv3DTranspose(64, kernel_size = (3,3,3), strides=2,  activation='relu', padding='same'),
            layers.Conv3D(1,           kernel_size = (3,3,3), activation='sigmoid', padding='same')
            # Ensure that the final layer restores the original input shape
        ])
        # Autoencoder
        autoencoder = models.Model(inputs=net.input, outputs=net.output)

        return autoencoder

    def custom_mse(self,y_true, y_pred):
        mask = tf.math.is_finite(y_true)
        y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
        y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))
        return tf.reduce_mean(tf.square(y_true - y_pred))


    def train(self,autoencoder, train_inputs, train_labels, 
              validation_inputs, validation_labels, 
              epochs=10, batch_size=32):
        autoencoder.compile(optimizer='adam', loss=self.custom_mse)
        
        # Train the model with both training and validation data
        history = autoencoder.fit(
            train_inputs, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(validation_inputs, validation_labels)
        )
        return history



if __name__=="__main__":
    input_shape = (112, 128, 112,1)

    ddn = DeepDenoiser()
    neuralnet = ddn.build_autoencoder(input_shape)
    neuralnet.summary()

# # Example usage
# input_shape = (m, l, k, channels)  # Define the input shape of your 3D data; channels might be 1 for grayscale
# autoencoder = build_autoencoder(input_shape)

# # Train the model (replace 'train_data' with your actual data)
# train_autoencoder(autoencoder, train_data, epochs=10, batch_size=32)