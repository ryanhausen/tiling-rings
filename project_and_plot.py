# Helpful resources ============================================================
# https://en.wikipedia.org/wiki/Stereographic_projection
# https://math.ucr.edu/~res/math153/history07d.pdf
# https://math.stackexchange.com/questions/1205927/how-to-calculate-the-area-covered-by-any-spherical-rectangle
# https://www.wolframcloud.com/objects/demonstrations/TangentPlaneToASphere-source.nb
# https://math.stackexchange.com/a/607434
# https://pubs.er.usgs.gov/publication/pp1395


import math
from collections import namedtuple
from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from rings import build_rings



class Point:
    d2r = np.pi / 180

    def __init__(
        self,
        xyz:np.ndarray=None,
        radec:np.ndarray=None,
        uv:np.ndarray=None
    ):
        self.xyz = xyz if xyz else Point.radec_to_xzy(radec)
        self.r = np.linalg.norm(self.xyz, ord=2)
        self.radec = radec if radec else Point.xyz_to_radec(xyz)
        self.uv = Point.radec_to_uv(self.radec)

    @property
    def x(self) -> float:
        return self.xyz[0]

    @property
    def y(self) -> float:
        return self.xyz[1]

    @property
    def z(self) -> float:
        return self.xyz[2]

    @property
    def ra(self) -> float:
        return self.radec[0]

    @property
    def dec(self) -> float:
        return self.radec[1]

    @property
    def u(self) -> float:
        return self.uv[0]

    @property
    def v(self) -> float:
        return self.uv[1]

    def dot(self, other:"Point") -> float:
        return self.xyz.dot(other.xyz)

    def magnitude(self) -> float:
        return np.norm(self.xyz, ord=2)

    def angular_distance(self, other:"Point") -> float:
        return np.arccos(self.dot(other)) / (self.r * other.r)

    def opposite(self) -> "Point":
        u, v = self.uv
        new_u = (u + np.pi)
        if new_u > 2*np.pi: new_u %= (2*np.pi)

        new_v = (v + np.pi)
        if new_v > np.pi: new_v %= np.pi

        return Point(uv=np.array([new_u, new_v]))

    @staticmethod
    def xyz_to_radec(xyz:np.ndarray) -> np.ndarray:
        x, y, z = xyz

        ra = np.arccos(z / np.linalg.norm(xyz, ord=2))
        dec = np.arctan2(y, x)

        return np.narray([ra, dec])

    @staticmethod
    def radec_to_uv(radec:np.ndarray) -> np.ndarray:
        return np.array([
            radec[0] * Point.d2r,
            np.abs(np.array(radec[1] * Point.d2r) - np.pi / 2)
        ])

    @staticmethod
    def radec_to_xzy(radec:np.ndarray, r:float=1) -> np.ndarray:
        uv = Point.radec_to_uv(radec)

        return np.array([
            r * np.cos(uv[0]) * np.sin(uv[1]),
            r * np.sin(uv[0]) * np.sin(uv[1]),
            r * np.cos(uv[1])
        ])



d2r = np.pi / 180
def main():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    a = 3.955

    u, v = get_uv(a_deg=a, semicircle=False)

    x, y, z = get_xyz(u, v)

    ax.scatter(x, y ,z, color="b", marker=".", alpha=0.1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])

    up, vp = u[0], v[0]

    a *= d2r

    us, vs = list(map(np.array, zip(*product([up-a/2, up+a/2], [vp-a/2, vp+a/2]))))
    min_u, max_u = us.min(), us.max()
    min_v, max_v = vs.min(), vs.max()
    n_points = 10
    us_range = np.linspace(min_u, max_u, num=n_points)
    vs_range = np.linspace(min_v, max_v, num=n_points)
    
    us, vs = np.meshgrid(us_range, vs_range)
    xs = np.zeros_like(us)
    ys = np.zeros_like(us)
    zs = np.zeros_like(us)
    for i, j in product(range(n_points), range(n_points)):
        x, y, z = get_xyz(us[i, j], vs[i, j])
        xs[i, j] = x
        ys[i, j] = y
        zs[i, j] = z

    ax.plot_surface(xs, ys, zs, color="r", alpha=0.25)


    us, vs = list(map(np.array, zip(*product([up-a, up+a], [vp-a, vp+a]))))
    min_u, max_u = us.min(), us.max()
    min_v, max_v = vs.min(), vs.max()
    n_points = 10
    us_range = np.linspace(min_u, max_u, num=n_points)
    vs_range = np.linspace(min_v, max_v, num=n_points)
    
    us, vs = np.meshgrid(us_range, vs_range)
    xs = np.zeros_like(us)
    ys = np.zeros_like(us)
    zs = np.zeros_like(us)
    for i, j in product(range(n_points), range(n_points)):
        x, y, z = get_xyz(us[i, j], vs[i, j])
        xs[i, j] = x
        ys[i, j] = y
        zs[i, j] = z

    ax.plot_surface(xs, ys, zs, color="g", alpha=0.25)


    # opp_u, opp_v = opposite(up, vp)
    # up_xyz = get_xyz(opp_u, opp_v)
    # ax.scatter(*get_xyz(opp_u, opp_v), color="g")
    # pole = np.array(get_xyz(opp_u, opp_v)) # 3,

    # R = 1
    # c = angular_distance([opp_u, opp_v], [us[0], vs[0]])
    # ρ = 2 * R * math.tan((c/2))
    # θ = np.pi - opp_u
    # k = np.arccos(c/2)**2


    # projected = [project(pole, tile_points[p,:]) for p in range(tile_points.shape[0])]
    # px, py, pz = list(zip(*projected))
    # ax.scatter(px, py, pz, color="g")

    # xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 100), np.linspace(-1.0, 1.0, 100))

    # dfdx = lambda _x: 2*_x
    # dfdy = lambda _y: 2*_y
    # dfdz = lambda _z: 2*_z

    # p = Point(
    #     np.cos(up) * np.sin(vp), # x
    #     np.sin(up) * np.sin(vp), # y
    #     np.cos(vp)               # z
    # )

    # zs = (-dfdx(p.x) * (xs - p.x) - dfdy(p.y) * (ys - p.y)) / dfdz(p.z) + p.z

    # ax.plot_wireframe(xs, ys, zs)

    # x = np.cos(us) * np.sin(vs)
    # y = np.sin(us) * np.sin(vs)
    # z = np.cos(vs)

    # ax.scatter(x, y, z, color="r", marker=".")





    plt.show()

def angular_distance(a, b):
    return math.acos(math.sin(a[1])*math.sin(b[1]) + math.cos(a[1])*math.cos(b[1])*math.cos(a[0]-b[0]))

def get_xyz(u, v, r=1):
    return (
        r * np.cos(u) * np.sin(v),
        r * np.sin(u) * np.sin(v),
        r * np.cos(v)
    )

def project(pole:np.ndarray, point:np.ndarray):
    # return pole + (point - pole) / (1 - point.dot(pole))
    return (point - pole) / ( - point.dot(pole))



def opposite(u, v):
    new_u = (u + np.pi)
    if new_u > 2*np.pi: new_u%=(2*np.pi)

    new_v = (v + np.pi)
    if new_v > np.pi: new_v%=np.pi

    return new_u, new_v


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

    u, v = np.array(u), np.abs(np.array(v) - np.pi / 2)

    if not semicircle:
        non_equator = v!=(np.pi / 2)
        flipped_v = v[non_equator] + np.pi
        flipped_u = u[non_equator]

        v = np.concatenate([v, flipped_v])
        u = np.concatenate([u, flipped_u])

    return u, v

if __name__=="__main__":
    main()