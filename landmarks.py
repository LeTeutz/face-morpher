import os
import sys
from scipy.spatial import Delaunay

spiga_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'spiga'))
sys.path.append(spiga_path)

from models.spiga.spiga.inference.config import ModelConfig
from models.spiga.spiga.inference.framework import SPIGAFramework

# code deleted, will be put back in January, 2024