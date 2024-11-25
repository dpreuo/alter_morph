import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d




def color_voronoi(
    positions,
    z,
    ax,
    minima=None,
    maxima=None,
    c_code="bwr",
    p_size=1,
    show_points=True,
    show_vertices=False,
    l_width=1,
    l_color="red",
    l_alpha=1,
    pol_alpha=1.0,
):
    """Wrapper around scipy voronoi plot for the lines, then color the regions according to given z.
    Most options can be controlled from this wrapper."""

    min_x, max_x = min(positions[:, 0]), max(positions[:, 0])
    min_y, max_y = min(positions[:, 1]), max(positions[:, 1])



    vor = Voronoi(positions)

    # If not given, find min/max values for normalization
    if minima is None:
        minima = min(z)
    if maxima is None:
        maxima = max(z)

    print(f"Min: {minima}, Max: {maxima}")

    # Normalize chosen colormap
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=c_code)

    for i, p_r in enumerate(vor.point_region):
        # Get corresponding region
        region = vor.regions[p_r]
        # Open regions (contain -1 as vertex) do not define a polygon
        if -1 not in region:
            # Define polygon from closed voronoi region
            polygon = [vor.vertices[v] for v in region]
            # Get ith color in z list corresponding to current point
            ax.fill(
                *zip(*polygon),
                alpha=pol_alpha,
                linewidth=0,  # No edges, voronoi_plot_2d takes care of that
                color=mapper.to_rgba(z[i]),
            )



    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    return ax
