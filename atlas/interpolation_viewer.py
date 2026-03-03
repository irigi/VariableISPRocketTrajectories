import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import LinearNDInterpolator


STATE_SOLVED = np.uint8(2)


def compute_edges_from_centers(values, log_spacing=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("values must be 1D with at least 2 elements")

    edges = np.empty(len(values) + 1, dtype=float)
    if log_spacing:
        if np.any(values <= 0):
            raise ValueError("log-spaced edges require strictly positive values")
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        edges[0] = values[0] ** 2 / edges[1]
        edges[-1] = values[-1] ** 2 / edges[-2]
    else:
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges



def robust_component_scales(values):
    """
    Compute a per-component normalization scale so that one component does not
    dominate the combined interpolation error merely because of its units/range.
    """
    q05 = np.nanpercentile(values, 5.0, axis=0)
    q95 = np.nanpercentile(values, 95.0, axis=0)
    scale = q95 - q05

    bad = ~np.isfinite(scale) | (scale <= 0)
    if np.any(bad):
        std = np.nanstd(values, axis=0)
        scale[bad] = std[bad]

    bad = ~np.isfinite(scale) | (scale <= 0)
    scale[bad] = 1.0
    return scale



def normalized_vector_error(y_true, y_pred, component_scale):
    """
    Unitless scalar error per point.

    error = sqrt(mean(((pred - true) / scale)^2)) over the 6 output components.
    """
    diff = (y_pred - y_true) / component_scale[None, :]
    return np.sqrt(np.mean(diff * diff, axis=1))



def build_interpolator(rho_grid, kappa_grid, theta_grid, data, train_mask):
    i_train, j_train, k_train = np.where(train_mask)
    if i_train.size < 4:
        raise ValueError(
            f"Need at least 4 solved odd-parity points for 3D interpolation, got {i_train.size}."
        )

    # Interpolate in transformed coordinates to respect the log-spaced axes.
    pts = np.column_stack([
        np.log10(rho_grid[i_train]),
        np.log10(kappa_grid[j_train]),
        theta_grid[k_train],
    ])
    vals = data[i_train, j_train, k_train, :]  # shape (n_points, 6)

    return LinearNDInterpolator(pts, vals, fill_value=np.nan), pts, vals



def evaluate_interpolator(interpolator, rho_grid, kappa_grid, theta_grid, data, eval_mask):
    i_eval, j_eval, k_eval = np.where(eval_mask)

    query_pts = np.column_stack([
        np.log10(rho_grid[i_eval]),
        np.log10(kappa_grid[j_eval]),
        theta_grid[k_eval],
    ])
    y_true = data[i_eval, j_eval, k_eval, :]
    y_pred = interpolator(query_pts)

    inside = np.all(np.isfinite(y_pred), axis=1)
    return i_eval, j_eval, k_eval, y_true, y_pred, inside



def summarize_errors(errors, label):
    if errors.size == 0:
        print(f"{label}: no valid points")
        return

    print(f"{label}:")
    print(f"  count   = {errors.size}")
    print(f"  mean    = {np.mean(errors):.6g}")
    print(f"  median  = {np.median(errors):.6g}")
    print(f"  p90     = {np.percentile(errors, 90):.6g}")
    print(f"  p95     = {np.percentile(errors, 95):.6g}")
    print(f"  max     = {np.max(errors):.6g}")



def visualize_error_slices(error_grid, rho_grid, kappa_grid, theta_grid,
                           nrows=3, ncols=4, figsize=(16, 10)):
    rho_edges = compute_edges_from_centers(rho_grid, log_spacing=True)
    kappa_edges = compute_edges_from_centers(kappa_grid, log_spacing=True)

    n_panels = nrows * ncols
    if len(theta_grid) <= n_panels:
        theta_indices = np.arange(len(theta_grid))
    else:
        theta_indices = np.linspace(0, len(theta_grid) - 1, n_panels, dtype=int)

    finite = error_grid[np.isfinite(error_grid) & (error_grid > 0)]
    if finite.size > 0:
        vmin = np.percentile(finite, 5)
        vmax = np.percentile(finite, 95)
        if not np.isfinite(vmin) or vmin <= 0:
            vmin = np.min(finite)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = np.max(finite)
        if vmax > vmin > 0:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for panel_idx, ax in enumerate(axes):
        if panel_idx >= len(theta_indices):
            ax.set_visible(False)
            continue

        k = theta_indices[panel_idx]
        z = error_grid[:, :, k].T

        mesh = ax.pcolormesh(
            rho_edges,
            kappa_edges,
            z,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rho = r_target / r_start")
        ax.set_ylabel("kappa")
        ax.set_title(f"theta[{k}] = {np.degrees(theta_grid[k]):.1f}°")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.92, pad=0.02)
        cbar.set_label("normalized interpolation error")

    finite_count = np.count_nonzero(np.isfinite(error_grid))
    fig.suptitle(
        "Interpolation error on held-out solved cells\n"
        f"(odd i+j+k solved cells used for training; finite error shown on {finite_count} cells)",
        fontsize=14,
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize interpolation error of a trajectory atlas using only odd-parity solved cells as "
                    "interpolation support."
    )
    parser.add_argument("-i", "--npz_path", help="Path to the trajectory atlas .npz file", required=True)
    args = parser.parse_args()

    bundle = np.load(args.npz_path)
    rho_grid = bundle["rho"]
    kappa_grid = bundle["kappa"]
    theta_grid = bundle["theta"]
    data = bundle["data"]
    state = bundle["state"]

    solved_mask = (state == STATE_SOLVED) & np.all(np.isfinite(data), axis=-1)

    ii, jj, kk = np.indices(state.shape)
    odd_mask = ((ii + jj + kk) % 2) == 1

    train_mask = solved_mask & odd_mask
    eval_mask = solved_mask & (~odd_mask)

    n_solved = int(np.count_nonzero(solved_mask))
    n_train = int(np.count_nonzero(train_mask))
    n_eval = int(np.count_nonzero(eval_mask))

    print(f"Solved cells total         : {n_solved}")
    print(f"Solved odd-parity training : {n_train}")
    print(f"Solved even-parity eval    : {n_eval}")

    if n_train < 4:
        raise RuntimeError("Not enough solved odd-parity points to build a 3D interpolator.")
    if n_eval == 0:
        raise RuntimeError("No solved even-parity points available for validation.")

    interpolator, train_pts, train_vals = build_interpolator(
        rho_grid, kappa_grid, theta_grid, data, train_mask
    )

    i_eval, j_eval, k_eval, y_true, y_pred, inside = evaluate_interpolator(
        interpolator, rho_grid, kappa_grid, theta_grid, data, eval_mask
    )

    print(f"Eval points inside interpolation hull : {int(np.count_nonzero(inside))}/{inside.size}")
    print(f"Eval points outside hull              : {int(np.count_nonzero(~inside))}/{inside.size}")

    component_scale = robust_component_scales(data[solved_mask])
    print("Per-component normalization scale     :", component_scale)

    error_grid = np.full(state.shape, np.nan, dtype=float)
    if np.any(inside):
        point_errors = normalized_vector_error(y_true[inside], y_pred[inside], component_scale)
        error_grid[i_eval[inside], j_eval[inside], k_eval[inside]] = point_errors
        summarize_errors(point_errors, "Held-out normalized interpolation error")
    else:
        print("No evaluation points are inside the convex hull of the odd solved points.")

    # Optional extra diagnostic: time-only relative error, because transfer time is often easy to interpret.
    if np.any(inside):
        t_true = y_true[inside, -1]
        t_pred = y_pred[inside, -1]
        t_scale = np.maximum(np.abs(t_true), 1e-12)
        time_rel_err = np.abs(t_pred - t_true) / t_scale
        summarize_errors(time_rel_err, "Held-out relative error in t_days")

    visualize_error_slices(error_grid, rho_grid, kappa_grid, theta_grid)


if __name__ == "__main__":
    main()
