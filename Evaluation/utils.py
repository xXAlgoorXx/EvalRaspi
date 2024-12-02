from typing import List, Generator, Optional, Tuple, Dict
from pathlib import Path
from functools import partial
import queue
from loguru import logger
import numpy as np
from PIL import Image
from hailo_platform import (HEF, VDevice,
                            FormatType, HailoSchedulingAlgorithm)

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import json

IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')

def loadJson(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    return data

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px):
    """
    n_px: input resolution of the network
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

