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
import matplotlib.pyplot as plt
from press import PressKey, ReleaseKey
from model import AutoDriveGRU


X = 580
Y = 750
WIDTH = 135
HEIGHT = 80
screen_position = (X, Y, X + WIDTH, Y + HEIGHT)
HIDDEN_SIZE = 480
NUM_LAYERS = 4
BATCH_SIZE = 3

W = 0x11
S = 0x1F
D = 0x20
A = 0x1E

EPOCH = 3
LR = 0.0001
VER = 1
SAVE_PATH = f"SelfDrivingGRU_lr_{LR}_ver_{VER}.pt"

LOAD_LR = None
LOAD_VER = None
LOAD_PATH = f"SelfDrivingGRU_lr_{LOAD_LR}_ver_{LOAD_VER}.pt"

DATA_PATH = "SelfDriving_dataset.pt"

loss_count = []


def plot_loss():
    plt.figure(1)
    plt.plot(loss_count)
    plt.title("Loss Graph")
    plt.xlabel("loop")
    plt.ylabel("loss")
    plt.show()


def grab_screen():
    screen = ImageGrab.grab(screen_position)
    screen = np.array(screen).transpose(2, 0, 1)
    screen = torch.from_numpy(screen)
    screen = screen.type(torch.float)
    return screen


def move(output, t=0.1):
    if output[0] == 1:
        ReleaseKey(S)
        PressKey(W)
        if output[2] == 1:
            PressKey(A)
            time.sleep(0.1)
            ReleaseKey(A)
        elif output[3] == 1:
            PressKey(D)
            time.sleep(0.1)
            ReleaseKey(D)
        else:
            time.sleep(t)

    elif output[1] == 1:
        ReleaseKey(W)
        PressKey(S)
        if output[2] == 1:
            PressKey(A)
            time.sleep(0.1)
            ReleaseKey(A)
        elif output[3] == 1:
            PressKey(D)
            time.sleep(0.1)
            ReleaseKey(D)
        else:
            time.sleep(t)

    elif output == [0, 0, 1, 0]:
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(A)
        time.sleep(t)
        ReleaseKey(A)

    elif output == [0, 0, 0, 1]:
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(D)
        time.sleep(t)
        ReleaseKey(D)


def check_finished():
    X = 640
    Y = 800
    WIDTH = 20
    HEIGHT = 10
    COLOR = (168, 84, 243)
    screen_position = (X, Y, X + WIDTH, Y + HEIGHT)

    screen = ImageGrab.grab(screen_position)
    screen = np.array(screen)
    if (screen != COLOR).all():
        return True

    return False


def train(save_path, load_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Driver = AutoDriveGRU(WIDTH, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE).to(device)
    if load_path:
        Driver.load_state_dict(torch.load(load_path))
    Driver.train()

    optimizer = torch.optim.AdamW(Driver.parameters(), lr=LR)
    mse_loss = torch.nn.MSELoss()
    training_data = torch.load(DATA_PATH)

    for e in range(EPOCH):
        print(f"--Epoch {e + 1}--")
        h = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)

        for i, data in enumerate(training_data):
            print(f"sample: {i + 1}")
            screen, target = data
            screen = screen.transpose(2, 0, 1).copy()
            screen = torch.from_numpy(screen)
            screen = screen.type(torch.float)
            target = torch.tensor(target).unsqueeze(0)

            output, h = Driver(screen, h)
            loss = mse_loss(output, target)
            print(f"loss: {loss.item()}")
            loss_count.append(loss.item())

            loss.backward()
            optimizer.step()
            print("\n")

        torch.save(Driver.state_dict(), save_path)

    print("--Finished Training--")
    plot_loss()


def evaluate(load_path):
    Driver = AutoDriveGRU(WIDTH, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE)
    Driver.load_state_dict(torch.load(load_path))
    Driver.eval()

    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("--Evaluation Start--")

    h = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)

    while True:
        if keyboard.is_pressed("q"):
            print("--Keyboard Break--")
            break

        if check_finished():
            print("--Arrived--")
            break

        screen = grab_screen()
        output, h = Driver(screen, h)
        output = output.squeeze()
        print("output:", output.tolist())
        output = [(x > 0.8).item() for x in output]
        output = [1 if x else 0 for x in output]

        move(output)

