# Helpful resources ============================================================
# https://math.ucr.edu/~res/math153/history07d.pdf
# https://math.stackexchange.com/questions/1205927/how-to-calculate-the-area-covered-by-any-spherical-rectangle
# https://www.wolframcloud.com/objects/demonstrations/TangentPlaneToASphere-source.nb
# https://math.stackexchange.com/a/607434


from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from rings import build_rings

d2r = np.pi / 180
Point = namedtuple("Point", ["x", "y", "z"])

def main():
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    a = 3.955
    u, v = get_uv(a_deg=a, semicircle=True)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    ax.scatter(x, y ,z, color="b", marker=".")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])

    up, vp = u[200], v[200]
    a *= d2r
    us, vs = list(map(np.array, zip(*product([up-a/2, up+a/2], [vp-a/2, vp+a/2]))))



    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 100), np.linspace(-1.0, 1.0, 100))

    dfdx = lambda _x: 2*_x
    dfdy = lambda _y: 2*_y
    dfdz = lambda _z: 2*_z

    p = Point(
        np.cos(up) * np.sin(vp), # x
        np.sin(up) * np.sin(vp), # y
        np.cos(vp)               # z
    )

    zs = (-dfdx(p.x) * (xs - p.x) - dfdy(p.y) * (ys - p.y)) / dfdz(p.z) + p.z

    ax.plot_wireframe(xs, ys, zs)

    x = np.cos(us) * np.sin(vs)
    y = np.sin(us) * np.sin(vs)
    z = np.cos(vs)

    ax.scatter(x, y, z, color="r", marker=".")

    plt.show()


def get_uv_from_file():

    with open("rings-c.txt", "r") as f:
        lines = list(filter(
            lambda x: len(x) == 3,
            [list(map(float,l.strip().split())) for l in f.readlines()]
        ))

    v, u = zip(*list(map(lambda x: (x[1], x[2]), lines)))

    return np.array(u), np.array(v)

def get_uv(a_deg=3.955, semicircle=True):
    ring_data = build_rings(a_deg=a_deg)
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