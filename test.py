import torch
from torchvision import transforms
import numpy as np
from PIL import ImageGrab, Image
import cv2
import time
import os
import keyboard
import mouse
from random import shuffle
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


