"""
Feature importance and SHAP visualization for the trained LightGBM model.
"""

import matplotlib.pyplot as plt
import numpy as np
import lightgbm
import shap


def plot_importances(train_model, filename="feature_importance.jpg", max_features=20):
    """
    Plot and save LightGBM split-based feature importances (top N features).

    Parameters
    ----------
    train_model : LGBMRegressor
        Trained model.
    filename : str
        Output file path (supports any matplotlib-supported format).
    max_features : int
        Maximum number of features to display.
    """
    plot = lightgbm.plot_importance(train_model, max_num_features=max_features,
                                    figsize=(4, 8))
    fig = plot.get_figure()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Feature importance saved: {filename}")


def plot_shap(train_model, X_test, filename="shap_summary.jpg"):
    """
    Generate and save a SHAP summary beeswarm plot.

    Parameters
    ----------
    train_model : LGBMRegressor
        Trained model.
    X_test : pd.DataFrame
        Test feature matrix.
    filename : str
        Output file path.
    """
    explainer = shap.TreeExplainer(train_model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP summary saved: {filename}")


def plot_true_vs_pred(y_true, y_pred, filename="true_vs_pred.jpg",
                      title="True vs Predicted Committor"):
    """
    Scatter plot of true vs. predicted committor values.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted committor probabilities.
    filename : str
        Output file path.
    title : str
        Plot title.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, s=4, edgecolor="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x")
    plt.xlabel("True committor")
    plt.ylabel("Predicted committor")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"True vs. predicted saved: {filename}")
