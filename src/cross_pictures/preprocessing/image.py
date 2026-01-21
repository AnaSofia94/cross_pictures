import io
from PIL import Image, UnidentifiedImageError
import numpy as np

#normalize
mean_vec = np.array([0.485, 0.456, 0.406], dtype="float32")
stddev_vec = np.array([0.229, 0.224, 0.225], dtype="float32")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    #load image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        return np.zeros((1, 3, 224, 224), dtype="float32")
    image = image.resize((224, 224)) #resize
    image_array = np.asarray(image).astype("float32") / 255.0 #convert to numpy
    image_array = (image_array - mean_vec) / stddev_vec #normalize
    image_array = image_array.astype("float32")
    image_array = np.transpose(image_array, (2, 0, 1)) #HWC -> CHW
    image_array = np.expand_dims(image_array, axis=0) #add batch dimension
    return image_array

