import matplotlib.pyplot as plt
import numpy as np

def colorcet_cmap_to_rgb(cmap, reverse=False, count=256):
    """
    Samples a colorcet colormap and prints RGB values in {r, g, b} integers format.
    """

    for i in range(count):
        if reverse:
            r, g, b = hex_to_rgb_ints(cmap[count - 1 - i])
        else:
            r, g, b = hex_to_rgb_ints(cmap[i])
        print(f"{{{r}, {g}, {b}}},")

def hex_to_rgb_ints(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return (int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def matplotlib_cmap_to_rgb(cmap_name, count=256):
    """
    Samples a matplotlib colormap and prints RGB values in {r, g, b} integers format.
    """
    # Retrieve the colormap object
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Colormap '{cmap_name}' not found.")
        exit(1)

    # Generate 'count' values linearly spaced between 0.0 and 1.0
    vals = np.linspace(0, 1, count)
    
    # Get RGBA values (an array of shape [count, 4])
    rgba_colors = cmap(vals)

    # Iterate and print RGB components (ignoring alpha)
    for r, g, b, a in rgba_colors:
        # Convert to integers in the range [0, 255]
        r_int = np.round(r * 255).astype(int)
        g_int = np.round(g * 255).astype(int)
        b_int = np.round(b * 255).astype(int)
        print(f"{{{r_int}, {g_int}, {b_int}}},")

if __name__ == "__main__":
    matplotlib_cmap_to_rgb('gist_rainbow')