from keras.layers import TFSMLayer
import numpy as np

class MaskDetector:
    def __init__(self, model_path):
        # Load TensorFlow SavedModel as inference layer using Keras 3
        self.model = TFSMLayer(model_path, call_endpoint="serving_default")

    def predict(self, image):
        """
        Predict mask status on the input image.
        
        Parameters:
        - image: A preprocessed NumPy array of shape (1, height, width, channels)

        Returns:
        - Prediction result (typically a probability or label)
        """
        # Ensure image is in batch format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        prediction = self.model(image)
        return prediction
