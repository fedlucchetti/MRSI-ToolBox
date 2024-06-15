import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import copy
from scipy.ndimage import gaussian_filter, binary_dilation
from tools.debug import Debug
from tqdm import tqdm
from skimage.filters import threshold_otsu

debug = Debug()


class StopTrainingOnStagnation(tf.keras.callbacks.Callback):
    def __init__(self, patience=4):
        super(StopTrainingOnStagnation, self).__init__()
        self.patience = patience
        # To keep track of minimum loss and stagnation epochs
        self.best_loss = None
        self.wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss <= self.best_loss:
            self.best_loss = current_loss
            self.wait = 0  # reset wait if there is improvement
        else:
            self.wait += 1  # increment wait if no improvement
            if self.wait >= self.patience:
                self.model.stop_training = True
                print("\nStopping training because the validation loss has stagnated for", self.patience, "epochs.")


class DeepSmoother(object):
    def __init__(self) -> None:
        pass

    def build_autoencoder(self,input_shape):
        # Encoder
        net = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv3D(128, kernel_size = (3,3,3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2), padding='same'),
            layers.Conv3D(64, kernel_size = (3,3,3), activation='relu', padding='same'),
            layers.Conv3DTranspose(128, kernel_size = (3,3,3), strides=2,  activation='relu', padding='same'),
            layers.Conv3D(1,           kernel_size = (3,3,3), activation='sigmoid', padding='same')
            # Ensure that the final layer restores the original input shape
        ])
        # Autoencoder
        autoencoder = models.Model(inputs=net.input, outputs=net.output)

        return autoencoder



    def custom_masked_cross_entropy(self,y_true, y_pred):
        # Mask out entries where y_true is -1
        mask = tf.not_equal(y_true, -1)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        
        # Compute binary cross-entropy on the masked values
        return tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)


    def train(self,autoencoder, train_inputs, train_labels, 
              validation_inputs, validation_labels, 
              epochs=10, batch_size=32,lrate=1e-2):
        
   
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate), 
                            loss=tf.keras.losses.binary_crossentropy)
        
        # Train the model with both training and validation data
        stagnation_callback = StopTrainingOnStagnation(patience=4)
        history = autoencoder.fit(
            train_inputs, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(validation_inputs, validation_labels),
            callbacks=[stagnation_callback]
        )
        return history

    def normalize_image(self,images):

        scales = list()
        norm_img = np.zeros(images.shape)
        scales = list()
        for idi,image in enumerate(images):
            scales.append([np.nanmin(image),np.nanmax(image)])
            norm_img[idi] = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
        return norm_img,scales

    def rescale_image(self,images,scales):
        scaled_img = np.zeros(images.shape)
        for idi,image in enumerate(images):
            min_v,max_v = scales[idi]
            scaled_img[idi] = image * (max_v-min_v)+min_v
        return scaled_img


    
    def peak_conc_mask(self,img3D,brain_mask,n=3.5):
        noSpike_image3D = copy.deepcopy(img3D)

        threshold = np.mean(img3D[brain_mask==1]) + n * np.std(img3D[brain_mask==1])
        # threshold = threshold_otsu(img3D[brain_mask==1])
        # Create a mask where voxel values are greater than the threshold
        conc_mask = img3D > threshold
        
        noSpike_image3D[conc_mask] = np.mean(noSpike_image3D)
        # mean  = np.mean(noSpike_image3D)
        # std   = np.std(noSpike_image3D)
        # # noise = np.random.normal(loc=mean, scale=std, size=conc_mask.sum())
        # noise = np.random.uniform(low=0.0, high=mean,size=conc_mask.sum())
        # noSpike_image3D[conc_mask] = noise
        return noSpike_image3D,conc_mask
    
    def preprocess(self,images,brain_masks,normalization=False):
        if len(images.shape)==3:
            images=np.expand_dims(images, axis=0)
        
        noSpike_image4D = np.zeros(images.shape)
        spike_mask4D    = np.zeros(images.shape)

        for idi,image3D in enumerate(images):
            image,peak_mask      = self.peak_conc_mask(image3D,brain_masks[idi])
            noSpike_image4D[idi] = image
            spike_mask4D[idi]    = peak_mask

        scales=list()
        if normalization:
            noSpike_image4D,scales = self.normalize_image(noSpike_image4D)
        
        return noSpike_image4D,spike_mask4D,scales
    
    def post_process(self,images,scales,scale=True):
        if scale:
            reshaped_images = self.rescale_image(images,scales)         
        if reshaped_images.shape[0]==1:
            reshaped_images=reshaped_images.squeeze()
        return reshaped_images

    def filter(self,filter,image3D,mask3D):
        noSpike_image3D, spike_mask3D, scales = self.preprocess(image3D,
                                                    mask3D,
                                                    normalization=True)
        filtered,_, _              = self.proc(filter, 
                                            noSpike_image3D, 
                                            spike_mask3D, 
                                            mask3D, 
                                            sigma=2, 
                                            dilation_iterations=2)
        filtered = self.post_process(np.expand_dims(filtered,axis=0),scales)

        return filtered

    def __predict(self,autoencoder,inputs):
        # filtered_img4D = np.zeros(inputs.shape)
        outputs        = np.zeros(inputs.shape)

        for ido,input3D in enumerate(inputs):
            outputs[ido]                     = autoencoder.predict(inputs[ido:ido+1],verbose=0).squeeze()
        return outputs
    
    def __crop_and_replace(self,inputs,raw_outputs,spike_mask4D):
        filled_img4D = np.zeros(inputs.shape)
        for idi, input3D in enumerate(inputs):
            filled_img                     = copy.deepcopy(input3D)
            filled_img[spike_mask4D[idi]==1] = raw_outputs[idi,spike_mask4D[idi]==1]
            filled_img4D[idi]              = filled_img
        return filled_img4D

    def __smooth_and_blend_edges(self,original_img, filtered_img, spike_mask4D, sigma=2, dilation_iterations=5):
        """

        Smooths and blends the edges of the replaced parts in the filtered image more effectively.

        Parameters:
        - original_img: The original image array of shape (N, 112, 128, 112).
        - filtered_img: The image array after filtering/interpolation of the same shape as original_img.
        - spike_mask4D: A boolean mask array of the same shape as original_img, indicating high conc voxels.
        - sigma: Standard deviation for Gaussian blur, controlling the extent of smoothing.
        - dilation_iterations: Number of iterations for mask dilation to expand the smoothing region.

        Returns:
        - blended_img: The image array after smoothing and blending the edges of the replaced parts.
        """
        # Ensure the hole mask is boolean
        spike_mask4D = spike_mask4D.astype(bool)
        
        # Create an empty array to store the blended image
        blended_img = np.copy(original_img)
        
        # Iterate through each input in the batch
        for ido in range(original_img.shape[0]):
            # Dilate the hole mask to include a broader area around the holes
            dilated_mask = binary_dilation(spike_mask4D[ido], iterations=dilation_iterations)
            
            # Apply Gaussian blur to the entire filtered image
            blurred_filtered_img = gaussian_filter(filtered_img[ido].astype(float), sigma=sigma)
            
            # Apply the blurred image only within the dilated mask region
            blended_portion = blended_img[ido] * (~dilated_mask) + blurred_filtered_img * dilated_mask
            
            # Update the original image with the blended portion
            blended_img[ido] = blended_portion
        
        return blended_img

    def __proc(self,smoother, inputs, spike_mask4D, brain_masks, sigma=2, dilation_iterations=2):

        raw_outputs   = self.__predict(smoother,inputs)

        cropped_img4D = self.__crop_and_replace(inputs,raw_outputs,spike_mask4D)

        blended_img   = self.__smooth_and_blend_edges(inputs, cropped_img4D, spike_mask4D, 
                                                    sigma=sigma,
                                                    dilation_iterations=dilation_iterations)
        blended_img[brain_masks==0] = 0
        return blended_img, cropped_img4D, raw_outputs


    def proc(self,filter,signals4D,masks4D):
        noSpike_image, spike_mask, scales = self.preprocess(signals4D,
                                                            masks4D,
                                                            normalization=True)
        filtered,_, _   = self.__proc(filter, 
                                    noSpike_image, 
                                    spike_mask, 
                                    masks4D, 
                                    sigma=2, 
                                    dilation_iterations=2)
        filtered        = self.post_process(np.expand_dims(filtered,axis=0),scales)

        return filtered, spike_mask.squeeze()






if __name__=="__main__":
    input_shape = (32, 42, 22,1)

    ddn = DeepSmoother()
    neuralnet = ddn.build_autoencoder(input_shape)
    neuralnet.summary()
