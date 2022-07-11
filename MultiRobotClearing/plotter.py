import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import os


class Plotter:
    def __init__(self, N, env, obs):
        import torch
        env = torch.zeros((3 ,16, 16))
        plt.gca().cla()
        self.robot_radius = 20
        self.map_size = N
        self.max_x = []
        self.max_y = []

        self.visited = env[2]
        self.imgs = []

        self.robot_colors = {
            1: "red",
            2: "green",
            3: "blue",
            4: "purple"
        }
        self.robot_pos = {
            1: {"x": [0], "y": [0], "c": self.robot_colors[1]},
            2: {"x": [15], "y": [0], "c": self.robot_colors[2]},
            3: {"x": [0], "y": [15], "c": self.robot_colors[3]},
            4: {"x": [15], "y": [15], "c": self.robot_colors[4]},
        }
        self.obstacle_pos = {
            "x": [],
            "y": []
        }

        # obstacles
        for pos_y, i in enumerate(obs):
            for pos_x, j in enumerate(i):
                if j == 1:
                    self.obstacle_pos["x"].append(pos_x, )
                    self.obstacle_pos["y"].append(pos_y)

        # visited
        for pos_y, i in enumerate(env[2]):
            for pos_x, j in enumerate(i):
                self.max_x.append(pos_x)
                self.max_y.append(pos_y)

    def move(self, robot_id, robot_x, robot_y):
        self.visited[self.robot_pos[robot_id]["y"][0]][self.robot_pos[robot_id]["x"][0]] = robot_id
        self.robot_pos[robot_id]["x"] = [robot_x]
        self.robot_pos[robot_id]["y"] = [robot_y]

    def graph(self, ep, step):
        plt.gca().cla()
        plt.title(f"Episode {ep}  Step {step}")
        # making general graph
        plt.plot(self.max_x, self.max_y, "r+", color="grey")
        # plotting visited
        loc_a = np.where(self.visited==1.0)
        loc_b = np.where(self.visited==2.0)
        loc_c = np.where(self.visited==3.0)
        loc_d = np.where(self.visited==4.0)
        plt.plot(loc_a[0], loc_a[1], "r+", color=self.robot_colors[1])
        plt.plot(loc_b[0], loc_b[1], "r+", color=self.robot_colors[2])
        plt.plot(loc_c[0], loc_c[1], "r+", color=self.robot_colors[3])
        plt.plot(loc_d[0], loc_d[1], "r+", color=self.robot_colors[4])
        # plotting robots
        for i in self.robot_pos:
            plt.plot(self.robot_pos[i]["x"], self.robot_pos[i]["y"], ".k",
                     markersize=self.robot_radius, color=self.robot_pos[i]["c"])
        # plt.show()

        # plotting obstacles
        plt.plot(self.obstacle_pos["x"], self.obstacle_pos["y"], ".k", color="black")
        plt.axis("equal")

        try:
            os.mkdir("./graphs")
        except FileExistsError:
            pass
        try:
            os.mkdir("./graphs/Episode_{}".format(ep))
        except FileExistsError:
            pass
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        self.imgs.append(Image.open(buf))

    def to_gif(self, ep):
        self.imgs[0].save("./graphs/Episode_{}/final.gif".format(ep), format='GIF',
                           append_images=self.imgs[1:],
                           save_all=True,
                           duration=50, loop=0)
        self.imgs = []


if __name__ == '__main__':

    env = [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

           [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

           [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
    plot = Plotter(16, env, env[0])
    plot.graph(1, 1)
    exit()
    plot.to_gif(0)




