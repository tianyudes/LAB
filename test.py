# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
data_set = pd.read_csv('testfile.csv')