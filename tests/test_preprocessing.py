import numpy as np
from pathlib import Path

from cross_pictures.preprocessing.image import preprocess_image

def test_preprocess_image_returns_correct_shape_and_dtype():
    image_path=Path("tests/assets/cat.jpg")
    image_bytes = image_path.read_bytes()

    tensor = preprocess_image(image_bytes)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
