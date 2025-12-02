!pip install "numpy<2.0" ultralytics opencv-python-headless tqdm cvzone"

import os  # for interact with operating system
import shutil  # for copying moving deleting files
import glob  # for finding file matching pattern
import random  # for shuffling images before split
import subprocess
import xml.etree.ElementTree as ET  # for reading xml file, parsing and for xml to yolo format
from collections import Counter  # for counting classes
from collections import defaultdict  # for maps
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # To read actual image size
from tqdm import tqdm  # progress bar
from tqdm.notebook import tqdm as tqdm_notebook  # notebook-friendly progress bar
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from IPython.display import Video, display, FileLink
