import torch
import matplotlib.pyplot as plt

def asymmetric_cantor_pytorch(a, b, iterations):
    """
    Generate levels of an asymmetric Cantor set using PyTorch for parallelization.
    
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
    levels : list of PyTorch tensors
        levels[0] is the tensor with the initial interval [[0, 1]],
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
    
    # Use torch tensors for parallel computation
    levels = [torch.tensor([[0.0, 1.0]], dtype=torch.float32)]
    
    for i in range(iterations):
        cur_intervals = levels[-1]
        
        # Get parameters for current iteration
        ai = a_list[i] if i < len(a_list) else a_list[-1]
        bi = b_list[i] if i < len(b_list) else b_list[-1]
        
        # Check for invalid parameters
        if not (0 <= ai <= 1 and 0 <= bi <= 1 and ai + bi < 1):
            raise ValueError(f"Invalid a,b at iteration {i}: a={ai}, b={bi}. Require 0<=a,b<=1 and a+b<1.")
            
        # Calculate lengths of current intervals in parallel
        lengths = cur_intervals[:, 1] - cur_intervals[:, 0]
        
        # Generate new left intervals
        left_intervals = torch.stack([
            cur_intervals[:, 0],
            cur_intervals[:, 0] + ai * lengths
        ], dim=1)
        
        # Generate new right intervals
        right_intervals = torch.stack([
            cur_intervals[:, 1] - bi * lengths,
            cur_intervals[:, 1]
        ], dim=1)
        
        # Concatenate the new intervals for the next level
        next_level = torch.cat([left_intervals, right_intervals], dim=0)
        
        # Filter out intervals with non-positive length
        positive_length_mask = next_level[:, 1] > next_level[:, 0]
        next_level = next_level[positive_length_mask]
        
        levels.append(next_level)
    return levels

def plot_levels_pytorch(levels, figsize=(8, 6), linewidth=6):
    """
    Plot the list of interval levels from PyTorch tensors. levels[0] is bottom (y=0).
    """
    plt.figure(figsize=figsize)
    n = len(levels)
    for i, level in enumerate(levels):
        y = n - 1 - i  # plot top-down so first iteration is at top
        for interval in level:
            L, R = interval.tolist()
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
levels_pt = asymmetric_cantor_pytorch(a, b, iterations)
plot_levels_pytorch(levels_pt)

# Print final level details
final_intervals = levels_pt[-1]
print(f"Number of intervals at final level: {len(final_intervals)}")
print("First 10 intervals at the final level:")
print(final_intervals[:10])