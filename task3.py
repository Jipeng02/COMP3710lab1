# Asymmetric Cantor set — generation and plot
# This code generates an asymmetric Cantor-like set by keeping left and right pieces
# of relative lengths `a` and `b` (0 < a, b < 1, a + b < 1) at each iteration.
# It plots the intervals at each iteration (stacked vertically) so you can see the construction.

import matplotlib.pyplot as plt

def asymmetric_cantor(a, b, iterations):
    """
    Generate levels of an asymmetric Cantor set.
    
    Parameters
    ----------
    a : float or list/tuple of floats
        Fraction of each interval length kept on the left side at each iteration.
        If a single float, it is reused at every iteration. If an iterable, it should
        have length >= iterations and a[i] is used for iteration i.
    b : float or list/tuple of floats
        Fraction kept on the right side (same semantics as `a`).
    iterations : int
        Number of iterations (levels) to compute.
    
    Returns
    -------
    levels : list of lists of (left, right) tuples
        levels[0] is the list with the initial interval [(0,1)],
        levels[1] the intervals after the first removal, etc.
    """
    # Normalize input to call-by-iteration lists
    def expand_param(p):
        if hasattr(p, '__iter__') and not isinstance(p, (str, bytes)):
            return list(p)
        else:
            return [p] * iterations
    a_list = expand_param(a)
    b_list = expand_param(b)
    
    levels = [[(0.0, 1.0)]]
    for i in range(iterations):
        cur = levels[-1]
        next_level = []
        ai = a_list[i] if i < len(a_list) else a_list[-1]
        bi = b_list[i] if i < len(b_list) else b_list[-1]
        if not (0 <= ai <= 1 and 0 <= bi <= 1 and ai + bi < 1):
            raise ValueError(f"Invalid a,b at iteration {i}: a={ai}, b={bi}. Require 0<=a,b<=1 and a+b<1.")
        for (L, R) in cur:
            length = R - L
            left_interval = (L, L + ai * length)
            right_interval = (R - bi * length, R)
            # Only keep intervals with positive length
            if left_interval[1] > left_interval[0]:
                next_level.append(left_interval)
            if right_interval[1] > right_interval[0]:
                next_level.append(right_interval)
        levels.append(next_level)
    return levels

def plot_levels(levels, figsize=(8, 6), linewidth=6):
    """
    Plot the list of interval levels. levels[0] is bottom (y=0).
    """
    plt.figure(figsize=figsize)
    n = len(levels)
    for i, level in enumerate(levels):
        y = n - 1 - i  # plot top-down so first iteration is at top
        for (L, R) in level:
            plt.hlines(y, L, R, linewidth=linewidth)
    plt.ylim(-1, n)
    plt.xlim(-0.05, 1.05)
    plt.xlabel("x")
    plt.yticks([])
    plt.title("Asymmetric Cantor set construction (each horizontal line = a level)")
    plt.tight_layout()
    plt.show()

# Example 1: constant asymmetry at every iteration
a = 0.25   # keep 25% on the left
b = 0.10   # keep 10% on the right
iterations = 6
levels = asymmetric_cantor(a, b, iterations)
plot_levels(levels)

# Example 2: varying asymmetry across iterations (optional)
# a_seq = [0.3, 0.2, 0.25, 0.18, 0.22, 0.2]
# b_seq = [0.1, 0.12, 0.08, 0.15, 0.1, 0.12]
# levels_var = asymmetric_cantor(a_seq, b_seq, iterations=len(a_seq))
# plot_levels(levels_var)

# If you want the final set of interval endpoints as a flat list:
final_intervals = levels[-1]
print(f"Number of intervals at final level: {len(final_intervals)}")
final_intervals[:10]  # show up to first 10 intervals (displayed by the notebook output)