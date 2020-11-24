import torch
import torchvision
import cv2
import numpy as np
import time
from PIL import Image, ImageGrab
import keyboard
import mouse
import pyautogui
import os


save_path = "SelfDriving_dataset.pt"

if os.path.isfile(save_path):
    training_data = torch.load(save_path)
else:
    training_data = []

X = 580
Y = 750
WIDTH = 135
HEIGHT = 80
screen_position = (X, Y, X + WIDTH, Y + HEIGHT)


def get_training_data():
    for i in range(1, 4):
        print(i)
        time.sleep(1)
    print("Go")

    while True:
        screen = ImageGrab.grab(screen_position)
        screen = np.array(screen)
        key = [0, 0, 0, 0]

        if keyboard.is_pressed("w"):
            key[0] = 1
            if keyboard.is_pressed("a"):
                key[2] = 1
            elif keyboard.is_pressed("d"):
                key[3] = 1
            training_data.append([screen, key])

        if keyboard.is_pressed("s"):
            key[1] = 1
            if keyboard.is_pressed("a"):
                key[2] = 1
            elif keyboard.is_pressed("d"):
                key[3] = 1
            training_data.append([screen, key])

        if keyboard.is_pressed("a"):
            key[2] = 1
            training_data.append([screen, key])

        if keyboard.is_pressed("d"):
            key[3] = 1
            training_data.append([screen, key])

        if len(training_data) % 2000 == 0 and len(training_data) != 0:
            torch.save(training_data, save_path)

        if keyboard.is_pressed("q"):
            print(len(training_data))
            torch.save(training_data, save_path)
            print("saved break")
            break


if __name__ == "__main__":
    get_training_data()
