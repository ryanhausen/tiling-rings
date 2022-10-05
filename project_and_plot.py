# Helpful resources ============================================================
# https://math.ucr.edu/~res/math153/history07d.pdf
# https://math.stackexchange.com/questions/1205927/how-to-calculate-the-area-covered-by-any-spherical-rectangle



import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from rings import build_rings

d2r = np.pi / 180

def main():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    u, v = get_uv(False)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    ax.scatter(x, y ,z, color="b", marker=".")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1.2)
    # ax.set_aspect("equal")
    ax.set_box_aspect([1,1,1])

    plt.show()


def get_uv_from_file():

    with open("rings-c.txt", "r") as f:
        lines = list(filter(
            lambda x: len(x) == 3,
            [list(map(float,l.strip().split())) for l in f.readlines()]
        ))

    v, u = zip(*list(map(lambda x: (x[1], x[2]), lines)))

    return np.array(u), np.array(v)

def get_uv(semicircle=True):
    ring_data = build_rings()
    u, v = [], []

    for v_key in ring_data:
        for u_data in ring_data[v_key]:
            u.append(u_data * d2r)
            v.append(v_key * d2r)

    u, v = np.array(u), np.array(v) - np.pi / 2

    if not semicircle:
        non_equator = v!=(np.pi / 2)
        flipped_v = v[non_equator] + np.pi
        flipped_u = u[non_equator]

        v = np.concatenate([v, flipped_v])
        u = np.concatenate([u, flipped_u])

    return u, v

if __name__=="__main__":
    main()