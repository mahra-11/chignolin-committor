"""
Free energy surface and committor-colored contour plots.

Free energy is computed from the 2D probability density via:
    F = -kT * ln(P),   with kT = 0.593 kcal/mol at 300 K.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def plot_free_energy_2d(x, y, bins=50, xlabel="x", ylabel="y",
                        title="Free energy surface"):
    """
    Plot a 2D free energy surface from trajectory feature values.

    Parameters
    ----------
    x, y : array-like
        Feature values (one per frame).
    bins : int
        Number of histogram bins along each axis.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    H = H.T
    H[H == 0] = np.nan
    H = -0.593 * np.log(H)
    H = H - np.nanmin(H)

    plt.figure(figsize=(7, 5))
    plt.imshow(H, interpolation="nearest", origin="lower",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap="jet", aspect="auto")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Free Energy (kcal mol⁻¹)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_contours(df_features, trait1, trait2, committor_csv,
                  committor_col="Committor_prob"):
    """
    Overlay free energy contour lines on a committor probability color map.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature DataFrame (contains trait1 and trait2 columns).
    trait1, trait2 : str
        Column names for the x and y axes.
    committor_csv : str
        Path to the cluster probability CSV (must contain committor_col).
    committor_col : str
        Column name for the committor probability.
    """
    committor_series = pd.read_csv(committor_csv)[committor_col]
    top = pd.concat(
        [df_features[trait1], df_features[trait2], committor_series], axis=1
    )
    top.columns = [trait1, trait2, "Committor_prob"]
    top = top.replace("Null", np.nan).dropna()
    for col in top.columns:
        top[col] = pd.to_numeric(top[col], errors="coerce")
    top = top.dropna()

    x = top[trait1].values
    y = top[trait2].values
    committor = top["Committor_prob"].values.astype(np.float64)

    # Free energy surface
    bins = 50
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    H = H.T
    H[H == 0] = np.nan
    H = -0.593 * np.log(H)
    H = H - np.nanmin(H)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Interpolate committor onto a regular grid
    grid_x, grid_y = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    grid_committor = griddata((x, y), committor, (grid_x, grid_y), method="cubic")
    grid_committor = gaussian_filter(grid_committor, sigma=1.5)
    grid_committor = np.clip(grid_committor, 0, 1)

    plt.figure(figsize=(7, 5))
    contour = plt.contourf(grid_x, grid_y, grid_committor, levels=5,
                           cmap="bwr", vmin=0, vmax=1)
    plt.contour(X, Y, H, levels=20, colors="black", linewidths=1.0)

    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel("Committor Probability")
    plt.xlabel(trait1)
    plt.ylabel(trait2)
    plt.tight_layout()
    plt.show()


def plot_tdplot(x_values, *score_series, xlabel="Fraction of data", ylabel="R²"):
    """
    Plot R² vs. data fraction (learning curve / downsampling sweep).

    Parameters
    ----------
    x_values : array-like
        X axis values (e.g. fractions or cluster counts).
    *score_series : array-like
        One or more score arrays to overlay.
    """
    colors = ["b", "r", "g", "m", "c", "y", "k"]
    plt.figure(figsize=(8, 5))
    for i, scores in enumerate(score_series):
        plt.plot(x_values, scores, marker="o", linestyle="-",
                 color=colors[i % len(colors)], label=f"Model {i+1}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. {xlabel}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
