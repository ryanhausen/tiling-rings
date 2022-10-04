
from collections import OrderedDict
import numpy as np

def build_rings(a_deg=3.955):
    d2r = np.pi / 180

    half_a = a_deg / 2 * d2r # half_a in radians
    half_a_theta = np.arctan(half_a)

    ring_data = OrderedDict()

    d = 0
    ring = 0

    while d < np.pi / 2 - half_a_theta:
        dm = d - half_a_theta
        if d==0:
            dm=0

        m = int(np.ceil(np.pi * np.cos(dm) / half_a_theta))
        ip = 2 * np.pi / m
        dp = np.arctan(np.tan(d + half_a_theta) * np.cos(ip / 2))

        ring_data[d/d2r] = []
        for i in range(m):
            a = i * ip
            ring_data[d/d2r].append(a/d2r)

        d = half_a_theta + dp
        ring += 1

    return ring_data

def main_cport():
    d2r = np.pi / 180
    a_deg = 3.955 # cell size in degrees

    half_a = a_deg / 2 * d2r # half_a in radians
    half_a_theta = np.arctan(half_a)

    d = 0
    ring = 0

    f = open("rings-py.txt", "w")
    while d < np.pi / 2 - half_a_theta:
        dm = d - half_a_theta
        if d==0:
            dm=0

        m = int(np.ceil(np.pi * np.cos(dm) / half_a_theta))
        ip = 2 * np.pi / m
        dp = np.arctan(np.tan(d + half_a_theta) * np.cos(ip / 2))

        f.write(f"{ring} {m}\n")

        for i in range(m):
            a = i * ip
            f.write(f"{i} {d / d2r} {a / d2r}\n")

        d = half_a_theta + dp
        ring += 1

    f.close()


def compare():
    with open("rings-py.txt", "r") as f, open("rings-c.txt", "r") as g:
        py_lines = [list(map(float, l.strip().split())) for l in f.readlines()]
        c_lines = [list(map(float, l.strip().split())) for l in g.readlines()]

    for i in range(len(py_lines)):
        np.testing.assert_allclose(py_lines[i], c_lines[i])


if __name__=="__main__":
    main_cport()
    compare()
