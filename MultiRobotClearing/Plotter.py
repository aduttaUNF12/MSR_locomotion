import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import os


class Plotter:
    def __init__(self, N, env_, obs):
        import torch
        self.env_ = env_
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
        self.charger_pos = {
            "x": [],
            "y": []
        }

        # obstacles
        for pos_y, i in enumerate(obs):
            for pos_x, j in enumerate(i):
                if j == 1:
                    self.obstacle_pos["x"].append(pos_x, )
                    self.obstacle_pos["y"].append(pos_y)

        # charging stations
        for pos_y, i in enumerate(obs):
            for pos_x, j in enumerate(i):
                if j == 2:
                    self.charger_pos["x"].append(pos_x, )
                    self.charger_pos["y"].append(pos_y)

        # visited
        for pos_y, i in enumerate(env[2]):
            for pos_x, j in enumerate(i):
                self.max_x.append(pos_x)
                self.max_y.append(pos_y)

    def move(self, robot_id, robot_x, robot_y):
        self.visited[self.robot_pos[robot_id]["x"][0]][self.robot_pos[robot_id]["y"][0]] = robot_id
        self.robot_pos[robot_id]["x"] = [robot_x]
        self.robot_pos[robot_id]["y"] = [robot_y]

    def graph(self, ep, step):
        plt.gca().cla()
        plt.rc('axes', labelsize=50)
        plt.rc('xtick', labelsize=0)
        plt.rc('ytick', labelsize=0)
        f = plt.figure(dpi=40)
        f.set_figwidth(16)
        f.set_figheight(16)
        plt.ylabel(f"Final Solution  Step {step}")
        plt.tick_params(left=False, right=False)

        # making general graph
        plt.plot(self.max_x, self.max_y, "s", color="grey", markersize=50)
        plt.plot(self.max_x, self.max_y, "s", color="white", markersize=49)
        # plotting visited
        loc_a = np.where(self.visited==1.0)
        loc_b = np.where(self.visited==2.0)
        loc_c = np.where(self.visited==3.0)
        loc_d = np.where(self.visited==4.0)
        # plt.plot(loc_a[0], loc_a[1], "s", color="grey", markersize=50)
        plt.plot(loc_a[0], loc_a[1], "s", color=self.robot_colors[1], markersize=50)
        plt.plot(loc_b[0], loc_b[1], "s", color=self.robot_colors[2], markersize=50)
        plt.plot(loc_c[0], loc_c[1], "s", color=self.robot_colors[3], markersize=50)
        plt.plot(loc_d[0], loc_d[1], "s", color=self.robot_colors[4], markersize=50)

        # plotting robots
        for i in self.robot_pos:
            plt.plot(self.robot_pos[i]["x"], self.robot_pos[i]["y"], ".k",
                     markersize=100, color=self.robot_pos[i]["c"])
        # plt.show()

        # plotting obstacles
        plt.plot(self.obstacle_pos["x"], self.obstacle_pos["y"], "s", color="black", markersize=50)
        plt.plot(self.charger_pos["x"], self.charger_pos["y"], "s", color="red", markersize=30)
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
        # with open("./graphs/Episode_{}/s_{}.png".format(ep, step), "wb") as fout:
        #     plt.savefig(fout, format="png")

        plt.savefig(buf, format="png")
        plt.close(f)
        buf.seek(0)
        img = Image.open(buf)
        img = img.rotate(-90)
        # img.save("./graphs/Episode_{}/s_{}_.png".format(ep, step))

        self.imgs.append(img)

    def to_gif(self, ep):
        self.imgs[0].save("./graphs/Episode_{}/final.gif".format(ep), format='GIF',
                          append_images=self.imgs[1:],
                          save_all=True,
                          duration=30, loop=0)
        self.imgs = []



if __name__ == '__main__':

    # Gif to mp4

    # import moviepy.editor as mp
    #
    # clip = mp.VideoFileClip("./graphs/Episode_4017/final.gif")
    # clip.write_videofile("./graphs/Episode_4017/Episode4017.mp4")
    #
    # exit()


    import torch
    import re
    import time

    # 2 charge station
    # 1 obstacle

    """
    
            [[2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2],
            ]
    
    """

    env = [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
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

    # env[0] = np.flip(env[0])

    env[0][7][6] = 1
    env[0][7][7] = 1
    env[0][7][8] = 1
    env[0][8][6] = 1
    env[0][8][7] = 1
    env[0][8][8] = 1

    import json
    with open("robot_actions.json", "r") as fout:
        robot_data = json.load(fout)
    ep = 0
    robot_one_data = robot_data[f'{ep}']['1']

    # arr = np.loadtxt("./EpisodeCoordinates_RNN_map3.csv", delimiter="\n", dtype="str")
    total_time = time.time()
    plot = Plotter(16, env, env[0])
    for pos, data in enumerate(robot_one_data):
        plot.move(1, data[0], data[1])
        plot.move(2, robot_data[f"{ep}"]['2'][pos][0], robot_data[f"{ep}"]['2'][pos][1])
        plot.move(3, robot_data[f"{ep}"]['3'][pos][0], robot_data[f"{ep}"]['3'][pos][1])
        plot.move(4, robot_data[f"{ep}"]['4'][pos][0], robot_data[f"{ep}"]['4'][pos][1])
        plot.graph(ep, pos)
        if pos % 100 == 0:
            print(f"{pos/len(robot_one_data)}% complete (steps done {pos})")
    plot.to_gif(ep)
    # for a in arr[4010:]:
    #     plot = Plotter(16, env, env[0])
    #     start = time.time()
    #     temp_a = a
    #     ep = temp_a.split(",")[0]
    #     print(f"Graphing EP: {ep}")
    #     # res = a[1:]
    #     res = a.replace(ep, "")
    #     res = res.replace("\"", "")
    #     res = res[1:].replace("[", "").replace("]", "").replace("'", "")
    #     points = re.split(r'\),', res)
    #     if int(ep) == 4017:
    #         for step, point in enumerate(points):
    #             point = point.replace("(", "")
    #             x, y = point.split(",")
    #             x = int(x.replace(")", "").replace("(", "").replace(",", "").strip())
    #             y = int(y.replace(")", "").replace("(", "").replace(",", "").strip())
    #             plot.move(1, y, x)
    #             plot.graph(ep, step)
    #             if step % 100 == 0:
    #                 print(f"{step/len(points)}% complete (steps done {step})")
    #         print(f"Finished {ep}, took {(time.time()-start)/60} min")
    #         plot.to_gif(ep)
    #         exit(0)


    print(f"Total time took: {(time.time()-total_time)/60} min")





