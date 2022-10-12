from itertools import product
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

from rings import build_rings

D2R = np.pi / 180

@st.cache
def get_uv(a_deg=3.955, semicircle=True):
    ring_data = build_rings(a_deg=a_deg)
    u, v = [], []

    for v_key in ring_data:
        for u_data in ring_data[v_key]:
            u.append(u_data * D2R)
            v.append(v_key * D2R)

    u, v = np.array(u), np.abs(np.array(v) - np.pi / 2)

    if not semicircle:
        non_equator = v!=(np.pi / 2)
        flipped_v = v[non_equator] + np.pi
        flipped_u = u[non_equator]

        v = np.concatenate([v, flipped_v])
        u = np.concatenate([u, flipped_u])

    return u, v


def get_xyz(u, v, r=1):
    return (
        r * np.cos(u) * np.sin(v),
        r * np.sin(u) * np.sin(v),
        r * np.cos(v)
    )

def make_area(up, vp, a, scale_f):
    a_radians =  a * D2R
    inner_us, inner_vs = list(map(
        np.array,
        zip(*product(
            [up-scale_f(a_radians), up+scale_f(a_radians)],
            [vp-scale_f(a_radians), vp+scale_f(a_radians)]
        ))
    ))
    resolution = 10
    min_u, max_u = inner_us.min(), inner_us.max()
    min_v, max_v = inner_vs.min(), inner_vs.max()
    inner_us_range = np.linspace(min_u, max_u, num=resolution)
    inner_vs_range = np.linspace(min_v, max_v, num=resolution)
    plot_inner_us, plot_inner_vs = np.meshgrid(inner_us_range, inner_vs_range)
    xs = np.zeros_like(plot_inner_us)
    ys = np.zeros_like(plot_inner_us)
    zs = np.zeros_like(plot_inner_us)
    for i, j in product(range(resolution), range(resolution)):
        x, y, z = get_xyz(plot_inner_us[i, j], plot_inner_vs[i, j])
        xs[i, j] = x
        ys[i, j] = y
        zs[i, j] = z

    return xs, ys, zs

st.header("Examining Tiling Areas of The Sphere")

a = st.slider("Tile size 'a' in degrees", min_value=1.0, max_value=10.0, value=3.955)
semicircle = st.checkbox("Semicircle?", value=False)
marker_size = st.slider("Marker size", min_value=0.5, max_value=10.0, value=1.0)

u, v = get_uv(a_deg=a, semicircle=semicircle)
x, y, z = get_xyz(u, v)

# Highlighted area =============================================================
idx = st.slider("Selected index:", min_value=0, max_value=u.shape[0]-1, step=1)
up, vp = u[idx], v[idx]


# inner ========================================================================
# a_radians =  a * D2R
# inner_us, inner_vs = list(map(np.array, zip(*product([up-a_radians/2, up+a_radians/2], [vp-a_radians/2, vp+a_radians/2]))))
# resolution = 10
# min_u, max_u = inner_us.min(), inner_us.max()
# min_v, max_v = inner_vs.min(), inner_vs.max()
# inner_us_range = np.linspace(min_u, max_u, num=resolution)
# inner_vs_range = np.linspace(min_v, max_v, num=resolution)
# plot_inner_us, plot_inner_vs = np.meshgrid(inner_us_range, inner_vs_range)
# xs = np.zeros_like(plot_inner_us)
# ys = np.zeros_like(plot_inner_us)
# zs = np.zeros_like(plot_inner_us)
# for i, j in product(range(resolution), range(resolution)):
#     _x, _y, _z = get_xyz(plot_inner_us[i, j], plot_inner_vs[i, j])
#     xs[i, j] = _x
#     ys[i, j] = _y
#     zs[i, j] = _z
colors = [
    [0, "rgb(0, 255, 0)"],
    [1, "rgb(255, 0, 0)"],
]
# Highlighted area =============================================================

figure = px.scatter_3d(
    x = x,
    y = y,
    z = z,
    # name = "point"
    height=700,
)
figure.update_traces(marker_size=marker_size, marker=dict(color="blue"))

inner_xs, inner_ys, inner_zs = make_area(up, vp, a, lambda a: a/2)

figure.add_trace(go.Surface(
    x=inner_xs,
    y=inner_ys,
    z=inner_zs,
    cmin=0,
    cmax=1,
    surfacecolor=np.zeros_like(inner_xs),
    colorscale=colors,
    opacity=0.5,
))

outer_xs, outer_ys, outer_zs = make_area(up, vp, a, lambda a: a)

figure.add_trace(go.Surface(
    x=outer_xs,
    y=outer_ys,
    z=outer_zs,
    cmin=0,
    cmax=1,
    surfacecolor=np.ones_like(outer_xs),
    colorscale=colors,
    opacity=0.5,
))
# N = 50
# figure.add_trace(go.Mesh3d(x=(70*np.random.randn(N)),
#                    y=(np.random.randn(N)),
#                    z=(np.random.randn(N)),
#                    opacity=0.5,
#                    color='pink'
#                   ))


figure.update_layout(
    scene=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
    ),
    scene_aspectmode='cube'
)




st.plotly_chart(figure, use_container_width=True)
