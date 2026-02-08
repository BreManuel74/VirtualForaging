import numpy as np
import matplotlib.pyplot as plt

def slow_exponential(x, base=1.02, scale=235.75, offset=-231.75):
    """
    Create an exponential function that grows very slowly from y=4 at x=0 to y=50 at x=9.
    
    Parameters:
    x: input value or array
    base: exponential base (1.02 for very slow growth)
    scale: scaling factor (235.75 calculated to fit range)
    offset: vertical offset (-231.75 calculated to start at y=4)
    
    Returns:
    exponential value(s) ranging from 4 to 50 over x=0 to x=9
    """
    return scale * (base ** x) + offset

def slower_exponential(x, base=1.01):
    """
    Create an even slower exponential function from y=4 at x=0 to y=50 at x=9.
    Base = 1.01 for extremely slow growth.
    """
    scale = 46 / (base**9 - 1)  # ≈ 491.12
    offset = 4 - scale  # ≈ -487.12
    return scale * (base ** x) + offset

def ultra_slow_exponential(x, base=1.005):
    """
    Create an ultra-slow exponential function from y=4 at x=0 to y=50 at x=9.
    Base = 1.005 for extremely slow growth.
    """
    scale = 46 / (base**9 - 1)  # ≈ 1003.97
    offset = 4 - scale  # ≈ -999.97
    return scale * (base ** x) + offset

def modified_exponential(x, initial_base=1.02, transition_base=1.3, transition_point=9, scale=235.75, offset=-231.75):
    """
    Create an exponential function with different growth rates before and after x=9.
    Very slow growth from y=4 to y=50 for x <= 9, then faster growth.
    
    Parameters:
    x: input value or array
    initial_base: base for slow growth (x <= transition_point)
    transition_base: base for faster growth (x > transition_point) 
    transition_point: x value where growth rate changes (default 9)
    scale: scaling factor (235.75 to fit y=4 to y=50 range)
    offset: vertical offset (-231.75 to start at y=4)
    
    Returns:
    exponential value(s)
    """
    if isinstance(x, (list, np.ndarray)):
        result = np.zeros_like(x, dtype=float)
        for i, val in enumerate(x):
            if val <= transition_point:
                result[i] = scale * (initial_base ** val) + offset
            else:
                # Ensure continuity at transition point
                transition_value = scale * (initial_base ** transition_point) + offset
                excess = val - transition_point
                result[i] = transition_value * (transition_base ** excess)
        return result
    else:
        if x <= transition_point:
            return scale * (initial_base ** x) + offset
        else:
            # Ensure continuity at transition point
            transition_value = scale * (initial_base ** transition_point) + offset
            excess = x - transition_point
            return transition_value * (transition_base ** excess)

def ultra_slow_then_sharp(x, initial_base=1.003, transition_base=2.0, transition_point=9):
    """
    Create an exponential with ultra-slow initial growth then very sharp increase.
    Maintains y=4 at x=0 and y=50 at x=9.
    
    Parameters:
    x: input value or array
    initial_base: very slow base for x <= 9 (default 1.003)
    transition_base: sharp base for x > 9 (default 2.0)
    transition_point: where growth rate changes (default 9)
    """
    # Calculate scale and offset for slow phase
    scale = 46 / (initial_base**9 - 1)
    offset = 4 - scale
    
    if isinstance(x, (list, np.ndarray)):
        result = np.zeros_like(x, dtype=float)
        for i, val in enumerate(x):
            if val <= transition_point:
                result[i] = scale * (initial_base ** val) + offset
            else:
                # Start sharp phase from y=50
                excess = val - transition_point
                result[i] = 50 * (transition_base ** excess)
        return result
    else:
        if x <= transition_point:
            return scale * (initial_base ** x) + offset
        else:
            excess = x - transition_point
            return 50 * (transition_base ** excess)

def custom_piecewise_exponential(x, slow_base=1.002, sharp_base=1.8, transition_point=9):
    """
    Piecewise exponential with extremely slow start and sharp later growth.
    Even more dramatic than ultra_slow_then_sharp.
    
    Parameters:
    x: input value or array
    slow_base: extremely slow base for initial phase (default 1.002)
    sharp_base: sharp base for later phase (default 1.8) 
    transition_point: where growth rate changes (default 9)
    """
    # Calculate parameters for the slow phase to hit exactly y=4→50
    scale = 46 / (slow_base**transition_point - 1)
    offset = 4 - scale
    
    if isinstance(x, (list, np.ndarray)):
        result = np.zeros_like(x, dtype=float)
        for i, val in enumerate(x):
            if val <= transition_point:
                result[i] = scale * (slow_base ** val) + offset
            else:
                # Continuous sharp growth from y=50
                excess = val - transition_point
                result[i] = 50 * (sharp_base ** excess)
        return result
    else:
        if x <= transition_point:
            return scale * (slow_base ** x) + offset
        else:
            excess = x - transition_point
            return 50 * (sharp_base ** excess)

def plot_exponential_comparison():
    """
    Plot the fitted exponential functions with different growth rates.
    """
    x = np.linspace(0, 20, 100)
    
    # Different exponential growth rates
    y_current = slow_exponential(x)  # base=1.02
    y_slower = slower_exponential(x)  # base=1.01
    y_ultra_slow = ultra_slow_exponential(x)  # base=1.005
    y_modified = modified_exponential(x)  # two-phase growth
    y_ultra_sharp = ultra_slow_then_sharp(x)  # ultra-slow then very sharp
    y_custom = custom_piecewise_exponential(x)  # extremely slow then sharp
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, y_current, label='Base=1.02 (current)', linewidth=2, color='green')
    plt.plot(x, y_slower, label='Base=1.01 (slower)', linewidth=2, color='blue')
    plt.plot(x, y_ultra_slow, label='Base=1.005 (ultra-slow)', linewidth=2, color='purple')
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.7, label='x=9')
    plt.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.scatter([0, 9], [4, 50], color='red', s=50, zorder=5, label='Target points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Single-Phase Exponential Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y_modified, label='Modified (1.02→1.3)', linewidth=2, color='orange')
    plt.plot(x, y_ultra_sharp, label='Ultra-slow→Sharp (1.003→2.0)', linewidth=2, color='red')
    plt.plot(x, y_custom, label='Custom (1.002→1.8)', linewidth=2, color='magenta')
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.7, label='x=9')
    plt.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.scatter([0, 9], [4, 50], color='red', s=50, zorder=5, label='Target points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Two-Phase Exponential Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 200)
    
    plt.subplot(2, 2, 3)
    # Focus on x=0 to x=15 to see the differences more clearly
    x_focus = np.linspace(0, 15, 100)
    y_current_focus = slow_exponential(x_focus)
    y_modified_focus = modified_exponential(x_focus)
    y_ultra_sharp_focus = ultra_slow_then_sharp(x_focus)
    y_custom_focus = custom_piecewise_exponential(x_focus)
    
    plt.plot(x_focus, y_current_focus, label='Current (1.02)', linewidth=3, color='green')
    plt.plot(x_focus, y_modified_focus, label='Modified (1.02→1.3)', linewidth=3, color='orange')
    plt.plot(x_focus, y_ultra_sharp_focus, label='Ultra-slow→Sharp', linewidth=3, color='red')
    plt.plot(x_focus, y_custom_focus, label='Custom Piecewise', linewidth=3, color='magenta')
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.7, label='x=9')
    plt.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.scatter([0, 9], [4, 50], color='red', s=50, zorder=5, label='Target points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison: Slow Start vs Sharp Later Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 150)
    
    plt.subplot(2, 2, 4)
    # Zoomed view of first 10 values to see slow growth
    x_zoom = np.linspace(0, 12, 100)
    y_current_zoom = slow_exponential(x_zoom)
    y_ultra_sharp_zoom = ultra_slow_then_sharp(x_zoom)
    y_custom_zoom = custom_piecewise_exponential(x_zoom)
    
    plt.plot(x_zoom, y_current_zoom, label='Current (1.02)', linewidth=3, color='green')
    plt.plot(x_zoom, y_ultra_sharp_zoom, label='Ultra-slow→Sharp', linewidth=3, color='red')
    plt.plot(x_zoom, y_custom_zoom, label='Custom (1.002→1.8)', linewidth=3, color='magenta')
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.7, label='x=9')
    plt.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    plt.scatter([0, 9], [4, 50], color='red', s=50, zorder=5, label='Target points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Detailed View: Initial Growth Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def demo_first_fifty_values():
    """
    Demonstrate the values for the first 50 x values using different exponential growth rates.
    """
    x_values = np.arange(0, 20)  # Show 0 to 19 for comparison
    
    print("Exponential Functions Comparison: y=4 at x=0, y=50 at x=9")
    print("=" * 80)
    
    # Current exponential (base=1.02)
    y_current = slow_exponential(x_values)
    print("Current (base=1.02):")
    for i, (x, y) in enumerate(zip(x_values[:15], y_current[:15])):
        print(f"  x={x:2d}: y={y:8.2f}")
    
    print()
    
    # Ultra-slow then sharp
    y_ultra_sharp = ultra_slow_then_sharp(x_values)
    print("Ultra-slow then Sharp (1.003 → 2.0):")
    for i, (x, y) in enumerate(zip(x_values[:15], y_ultra_sharp[:15])):
        print(f"  x={x:2d}: y={y:8.2f}")
    
    print()
    
    # Custom piecewise
    y_custom = custom_piecewise_exponential(x_values)
    print("Custom Piecewise (1.002 → 1.8):")
    for i, (x, y) in enumerate(zip(x_values[:15], y_custom[:15])):
        print(f"  x={x:2d}: y={y:8.2f}")
    
    print()
    print("Notice how the new functions have much slower growth 0→9, then sharp increases after x=9")
    print("All functions maintain y=4.00 at x=0 and y=50.00 at x=9")

if __name__ == "__main__":
    # Run demonstration
    demo_first_fifty_values()
    
    # Create plots
    plot_exponential_comparison()
