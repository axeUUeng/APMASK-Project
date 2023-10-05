import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def mutiplyGauss (m1 , s1 , m2 , s2):
    s = 1 / (1/s1 + 1/s2)
    m = (m1/s1 + m2/s2) * s
    return m, s


def divideGauss (m1 , s1 , m2 , s2):
    m, s = mutiplyGauss(m1 , s1 , m2 , -s2)
    return m, s

def truncGaussMM (a, b, m0 , s0):
    a_scaled , b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = stats.truncnorm.mean(a_scaled , b_scaled , loc=m0 , scale=np.sqrt(s0))
    s = stats.truncnorm.var(a_scaled , b_scaled , loc=m0 , scale=np.sqrt(s0))
    return m, s

mu1 = 900
sigma1 = 200

mu2 = 1000
sigma2 = 50

beta_inv = 50

y= -1


mt, st = truncGaussMM(0, np.inf, mu1-mu2 , beta_inv + sigma1 + sigma2) if y==1 else  truncGaussMM(-np.inf, 0, mu1-mu2 , beta_inv + sigma1 + sigma2)
#mt += mu1 - mu2
m6, s6 = divideGauss(mt , st , mu1-mu2 , beta_inv + sigma1 + sigma2)

m71, m72 = m6 + mu2, m6 + mu1
s71, s72 = beta_inv + sigma2 + s6, beta_inv + sigma1 + s6

m1_star, s1_star = mutiplyGauss(mu1, sigma1, m71, s71)
m2_star, s2_star = mutiplyGauss(mu2, sigma2, m72, s72)

