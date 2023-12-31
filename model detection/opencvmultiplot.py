import cv2
import numpy as np


class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [
            (255, 0, 0),
            (0, 250, 0),
            (0, 0, 250),
            (0, 255, 250),
            (250, 0, 250),
            (250, 250, 0),
            (200, 100, 200),
            (100, 200, 200),
            (200, 200, 100),
        ]
        self.color = []
        self.val = []
        self.plot = np.ones((self.height, self.width, 3)) * 255

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])

    def multiplot(self, val, label="plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    def show_plot(self, label):
        self.plot = np.ones((self.height, self.width, 3)) * 255
        cv2.line(
            self.plot,
            (0, int(self.height / 2)),
            (self.width, int(self.height / 2)),
            (0, 255, 0),
            1,
        )
        for i in range(len(self.val) - 1):
            for j in range(len(self.val[0])):
                cv2.line(
                    self.plot,
                    (i, int(self.height / 2) - self.val[i][j]),
                    (i + 1, int(self.height / 2) - self.val[i + 1][j]),
                    self.color[j],
                    1,
                )

        cv2.imshow(label, self.plot)
        cv2.waitKey(10)
