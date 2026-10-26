"""
CV2090 | Monte Carlo simulation - 1 of 3
Area of a circle by throwing random points.

Circle: radius 0.5, center (0.5, 0.5), sitting inside the unit square.
Exact area = pi/4 = 0.785398...

Run:  python mcs1_circle_area.py
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Task 1: throw N points into the unit square, mark the ones inside
# -------------------------------------------------------------------
def throw_points(N, rng):
    """Return x, y, and q, where q[i] = 1 if point i is inside the circle."""
    x = rng.random(N)
    y = rng.random(N)
    q = ((x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.5 ** 2)
    return x, y, q.astype(float)


# -------------------------------------------------------------------
# Task 2: running estimate of the area and its 95% confidence interval
# -------------------------------------------------------------------
def running_estimate(q):
    """Area estimate after 1, 2, ... N throws, with 95% CI."""
    n = np.arange(1, len(q) + 1)          # 1, 2, 3, ... N
    mean_q = np.cumsum(q) / n             # running mean of q  -> area estimate
    mean_q2 = np.cumsum(q ** 2) / n       # running mean of q^2
    se = np.sqrt((mean_q2 - mean_q ** 2) / n)   # standard error of the mean
    return mean_q, mean_q - 1.96 * se, mean_q + 1.96 * se


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def main():
    N = 1000
    rng = np.random.default_rng(2026)     # seed, so everyone gets the same numbers

    x, y, q = throw_points(N, rng)
    area, lo, hi = running_estimate(q)

    print(f"N          = {N}")
    print(f"area (MCS) = {area[-1]:.4f}")
    print(f"area (exact) = {np.pi / 4:.4f}")
    print(f"95% CI     = [{lo[-1]:.4f}, {hi[-1]:.4f}]")
    print(f"pi         = {4 * area[-1]:.4f}")

    # Figure 1: the points, inside in red and outside in blue
    plt.figure()
    plt.scatter(x[q == 0], y[q == 0], s=8, c="b")
    plt.scatter(x[q == 1], y[q == 1], s=8, c="r")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    # Figure 2: how the estimate settles down as N grows
    plt.figure()
    plt.plot(area, "b-", label="MCS estimate")
    plt.plot(lo, "b--", label="95% CI")
    plt.plot(hi, "b--")
    plt.axhline(np.pi / 4, color="r", label="exact")
    plt.ylim(0.5, 1.0)
    plt.xlabel("N")
    plt.ylabel("area")
    plt.legend()

    plt.show()


main()
