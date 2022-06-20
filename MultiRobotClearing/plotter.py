import matplotlib.pyplot as plt
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
        for pos_y, i in enumerate(self.visited):
            for pos_x, j in enumerate(i):
                if j == 1:
                    plt.plot(pos_x, pos_y, "r+", color=self.robot_colors[1])
                elif j == 2:
                    plt.plot(pos_x, pos_y, "r+", color=self.robot_colors[2])
                elif j == 3:
                    plt.plot(pos_x, pos_y, "r+", color=self.robot_colors[3])
                elif j == 4:
                    plt.plot(pos_x, pos_y, "r+", color=self.robot_colors[4])

        # plotting robots
        for i in self.robot_pos:
            plt.plot(self.robot_pos[i]["x"], self.robot_pos[i]["y"], ".k",
                     markersize=self.robot_radius, color=self.robot_pos[i]["c"])
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

        plt.savefig("./graphs/Episode_{}/{}.png".format(ep, step))

    def to_gif(self, ep):
        import glob
        from PIL import Image
        frames = []
        imgs = sorted(glob.glob("./graphs/Episode_{}/*.png".format(ep)), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        frames[0].save("./graphs/Episode_{}/final.gif".format(ep), format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=50, loop=0)



# if __name__ == '__main__':
#
#     env = [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
#
#            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
#
#            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
#     plot = Plotter(16, env)
#     plot.to_gif(0)
#
#


