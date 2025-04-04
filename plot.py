import math
import matplotlib.pyplot as plt
import numpy as np
def calculate_angle(vector1, vector2):
    """
    计算两个向量之间的夹角（弧度）。
    :param vector1: 向量1，包含系数 (a, b, c)。
    :param vector2: 向量2，包含系数 (a, b, c)。
    :return: 两个向量之间的夹角（弧度）。
    """
    vector1 = vector1[:3]
    vector2 = vector2[:3]
    angle = math.acos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    return angle
image_counter = 0  # 用于保存图片时的计数
def plot_planes_and_points(plane_equations, points, axis_range=(0, 32)):
    """
    绘制空间中的平面和点云。
    :param plane_equations: 平面方程的系数 (a, b, c, d)，每个平面用一个元组表示。
    :param points: 点云数据，形状为 (n, 3)，包含 n 个点的坐标 (x, y, z)。
    :param axis_range: 坐标轴范围，默认为 (0, 32)。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 创建网格用于平面绘制
    x = np.linspace(axis_range[0], axis_range[1], 2000)
    y = np.linspace(axis_range[0], axis_range[1], 2000)
    x, y = np.meshgrid(x, y)
    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=1)
    should_plot = [True] * len(plane_equations)
    for i in range(len(plane_equations)):
        for j in range(i + 1, len(plane_equations)):
            if calculate_angle(plane_equations[i], plane_equations[j]) < math.pi / 6:
                should_plot[j] = False
    for i, plane in enumerate(plane_equations):
        if not should_plot[i]:
            continue
        a, b, c, d = plane
        z = (-a * x - b * y - d) / c
        mask = (z >= axis_range[0]) & (z <= axis_range[1])
        z[~mask] = np.nan
        ax.plot_surface(x, y, z, alpha=0.3, color='g')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axis('off')
    ax.view_init(elev=45, azim=-60)
    ax.set_xlim(axis_range[0], axis_range[1])
    ax.set_ylim(axis_range[0], axis_range[1])
    ax.set_zlim(axis_range[0], axis_range[1])
    global image_counter
    image_counter += 1
    plt.savefig(f'output/{image_counter}.png')
    # plt.show()
if __name__ == '__main__':
    plane_equations = [(3, 3, 3, 3), (1, 2, 3, 4)]
    points = np.random.uniform(-10, 10, size=(100, 3))  
    plot_planes_and_points(plane_equations, points, axis_range=(-10, 10))
