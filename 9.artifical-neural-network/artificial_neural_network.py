import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from

print(tf.__version__)

dataset = pd.read_csv('9.artifical-neural-network/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
