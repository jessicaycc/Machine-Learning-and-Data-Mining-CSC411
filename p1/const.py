import os
import numpy as np
from PIL import Image

LORR = KRIS = 0
PERI = FRAN = 1
ANGI = AMER = 2
ALEC = DANI = 3
BILL = GERA = 4
STEV = MICH = 5

DATA_NAME = 0
DATA_URL = 3
DATA_BBOX = 4
DATA_HASH = 5
DATA_SIZE = (32, 32)
DATA_SET_RATIO = (100, 10, 10)

VEC_SIZE = 1024
MAX_ITER = 15000
TSH_HOLD = 0.5
LRN_RATE = 0.005
DIV = 1e-5
EPS = 1e-5
