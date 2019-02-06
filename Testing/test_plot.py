from Experiments.helper.plots import plot_score_comparisons
import numpy as np


x = np.linspace(0,100,100)
y1 = np.sin(x/6*np.pi)
error1 = np.random.normal(0.1, 0.02, size=y1.shape) + 10
y1 += np.random.normal(0, 0.1, size=y1.shape)

y2 = np.cos(x/6*np.pi)  + np.sin(x/3*np.pi)  
error2 = np.random.rand(len(y2)) * 10
y2 += np.random.normal(0, 0.1, size=y2.shape)

plot_score_comparisons('/c/Users/marce/Desktop/', x, y1, y2, error1, error2,45, 55)