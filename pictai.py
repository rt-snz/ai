#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, os
import numpy as np
import pickle

import tkinter
from PIL import Image, ImageDraw, ImageOps
from datetime import datetime


# size
window_width = 280
window_height = 280
canvas_width = 28 * 10
canvas_height = 28 * 10

button_width = 5
button_height = 1

class PictAI():

    def __init__(self):
        self.window = self.create_window()

        # set canvas
        self.image1 = Image.new("RGB", (window_width, window_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image1)

        # set param
        with open("sample_weight.pkl", 'rb') as f:
             self.network = pickle.load(f)

    def run(self):
        self.window.mainloop()

    def create_window(self):
        window = tkinter.Tk()

        # canvas frame
        canvas_frame = tkinter.LabelFrame(window, bg="white",text="手書文字認識",width=window_width, height=window_height,relief='groove', borderwidth=4)
        canvas_frame.pack(side=tkinter.LEFT)

        self.canvas = tkinter.Canvas(canvas_frame, bg="white",width=canvas_width, height=canvas_height)
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)

        judge_button = tkinter.Button(canvas_frame, text="判定", width=button_width, height=button_height, command=self.judge)
        judge_button.pack(side=tkinter.LEFT)

        clear_button = tkinter.Button(canvas_frame, text="クリア", width=button_width, height=button_height, command=self.clear)
        clear_button.pack(side=tkinter.LEFT)


        return window

    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y

    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,width=10,tag="draw")
        self.draw.line(((self.sx, self.sy), (event.x, event.y)), (255, 255, 255), 10)

        self.sx = event.x
        self.sy = event.y

    def clear(self):
        self.canvas.delete("draw")

        self.image1 = Image.new("RGB", (window_width, window_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image1)

    def judge(self):
        now =datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')

        fileName = "pic_" + timestamp + ".png"
        self.image1.save(fileName)

        input_image = Image.open(fileName)
        gray_image = ImageOps.grayscale(input_image) # [ R G B ] => [ 0 - 255 ]

        resize_image = np.array(gray_image.resize((28, 28)).getdata())

        y = self.predict(resize_image)
        print(y)
        p = np.argmax(y)
        print (p)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def predict(self, x):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)

        return y

def main():
    PictAI().run()

if __name__ == '__main__':
    main()
