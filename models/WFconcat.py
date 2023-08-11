
import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept

from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box




class WF_Concat(nn.Module):
    def __init__(self, dimension=1):
        super(WF_Concat, self).__init__()
        self.d = dimension
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x): 
        if len(x) == 2:
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1]]
        elif len(x) == 3: 
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]

       
        return torch.cat(x, self.d)  
