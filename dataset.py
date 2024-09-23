from copy import deepcopy

import numpy as np
import cv2
import random

import torch
import torchvision
from functools import partial
import dill
from scipy.stats import norm


def evaluation_loss(outputs, y):
    y_pdf = norm.pdf(np.linspace(0, 1, 100), loc=y[0], scale=y[1])
    x_pdf = norm.pdf(np.linspace(0, 1, 100), loc=outputs[0], scale=outputs[1])

    return np.sum(np.power(x_pdf - y_pdf, 2))


train = np.random.uniform(0, 1, (10000, 2))
train[:, 1] *= 0.1
test = np.random.uniform(0, 1, (1000, 2))
test[:, 1] *= 0.1

eval = []
tt = []
for ii in range(1000):
    eval.append((torch.from_numpy(test[ii, :]).float(), deepcopy(partial(evaluation_loss, y=test[ii, :]))))

for ii in range(10000):
    mu, sigma = train[ii, :]
    cdf = norm.cdf(np.linspace(0, 1, 100), loc=mu, scale=sigma)
    xx = np.interp(np.random.uniform(0, 1, 100), cdf, np.linspace(0, 1, 100))

    tt.append((torch.from_numpy(xx).float(), torch.from_numpy(train[ii, :]).float()))
#test.append((torch.from_numpy(inverse_sampling(y)).float(), deepcopy(partial(evaluation_loss, y=y))))

dill.dump(tt, open('data/task4/train.pkl', 'wb'))
dill.dump(eval, open('data/task4/test.pkl', 'wb'))

exit()
with open('data/task4/train/labels.csv', 'w') as fp:
    keys = list(thetas.keys())
    for key in keys[:-1]:
        fp.write(f'{key}:{thetas[key]},')
    fp.write(f'{keys[-1]}:{thetas[keys[-1]]}')
exit()



dataset = torchvision.datasets.MNIST("data", download=False, train=False)

for idx, item in enumerate(dataset):
    img, label = item
    img = np.array(img)

    cv2.imwrite(f"data/test/{label}/test{idx:05d}.png", img)
