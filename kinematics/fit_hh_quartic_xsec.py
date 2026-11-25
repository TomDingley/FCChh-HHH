#!/usr/bin/env python3
"""Fit the HH monomials to plot_hh_quartic_xsec.py's CSV and plot smooth surfaces.

The basis follows arXiv:2402.03463, Eq. (2.1); shifting kappa -> kappa-1 spans
exactly the same biquadratic space. Coefficients here use UNshifted kappas.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Squaring A_box + k3*A_tri + k4*A_4 + k3*k4*A_34 gives these nine terms.
POWERS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)]
TERMS = ['1', 'k3', 'k4', 'k3^2', 'k3*k4', 'k4^2', 'k3^2*k4', 'k3*k4^2', 'k3^2*k4^2']


def design_matrix(k3, k4):
    k3, k4 = np.broadcast_arrays(k3, k4)
    return np.stack([k3**i * k4**j for i, j in POWERS], axis=-1)


def fit_polynomial(points, values, errors):
    """Absolute-error weighted least squares; SVD avoids normal-equation inversion."""
    if len(values) <= len(POWERS):
        raise ValueError('need more than nine points for a fit and residual diagnostics')
    if not all(np.all(np.isfinite(x)) for x in (points, values, errors)) or np.any(errors <= 0):
        raise ValueError('fit requires finite data and strictly positive header uncertainties')
    matrix = design_matrix(points[:, 0], points[:, 1])
    weighted = matrix / errors[:, None]
    # Scale the columns to keep mixed k3/k4 powers numerically well conditioned.
    scales = np.linalg.norm(weighted, axis=0)
    if np.any(scales == 0):
        raise ValueError('grid does not constrain all nine monomials')
    u, singular, vt = np.linalg.svd(weighted / scales, full_matrices=False)
    if singular[-1] <= singular[0] * max(weighted.shape) * np.finfo(float).eps:
        raise ValueError('grid does not constrain all nine monomials')
    coeff = (vt.T @ ((u.T @ (values / errors)) / singular)) / scales
    covariance = ((vt.T / singular**2) @ vt) / np.outer(scales, scales)
    predicted = matrix @ coeff
    residual = values - predicted
    pulls = residual / errors
    fractional = np.divide(residual, values, out=np.full_like(values, np.nan), where=values != 0)
    summary = dict(points=len(values), ndof=len(values) - len(coeff),
                   chi2=float(pulls @ pulls),
                   chi2_per_ndof=float(pulls @ pulls) / (len(values) - len(coeff)),
                   rms_relative_percent=float(100 * np.sqrt(np.nanmean(fractional**2))),
                   max_abs_relative_percent=float(100 * np.nanmax(np.abs(fractional))),
                   max_abs_pull=float(np.max(np.abs(pulls))),
                   scaled_condition_number=float(singular[0] / singular[-1]))
    return coeff, covariance, predicted, summary


def interpolate(points, values, x, y):
    """Linear interpolation of sigma, with no extrapolation outside the convex hull."""
    return LinearNDInterpolator(points, values, rescale=True)(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='CSV from plot_hh_quartic_xsec.py')
    parser.add_argument('--outdir', type=Path, default=Path('hh_quartic_fit'))
    parser.add_argument('--grid', type=Path, help='Optional requested coupling list to show the full planned axes')
    parser.add_argument('--k3-range', nargs=2, type=float, metavar=('MIN', 'MAX'))
    parser.add_argument('--k4-range', nargs=2, type=float, metavar=('MIN', 'MAX'))
    parser.add_argument('--vmin', type=float, help='Colour scale lower limit in pb')
    parser.add_argument('--vmax', type=float, help='Colour scale upper limit in pb')
    parser.add_argument('--extrapolate', action='store_true', help='Extend the polynomial fit over all of --grid')
    parser.add_argument('--log', action='store_true', help='Logarithmic cross-section colours')
    args = parser.parse_args()
    if args.extrapolate and args.grid is None and not (args.k3_range and args.k4_range):
        parser.error('--extrapolate requires --grid or both --k3-range and --k4-range')
    for limits in (args.k3_range, args.k4_range):
        if limits and (not np.all(np.isfinite(limits)) or limits[0] >= limits[1]):
            parser.error('plot ranges must be finite and increasing')
    if args.vmin is not None and args.vmax is not None and args.vmin >= args.vmax:
        parser.error('--vmin must be smaller than --vmax')
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    points = np.array([[float(r['k3']), float(r['k4'])] for r in rows])
    values = np.array([float(r['xsec_pb']) for r in rows])
    errors = np.array([float(r['error_pb']) for r in rows])
    if len(set(map(tuple, points))) != len(points):
        parser.error('CSV has duplicate coupling points; aggregate LHE seeds first')
    try:
        coeff, covariance, predicted, summary = fit_polynomial(points, values, errors)
    except ValueError as exc:
        parser.error(str(exc))
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = dict(basis='sigma_pb = sum(c * k3**i * k4**j); unshifted kappas',
                  reference='https://arxiv.org/html/2402.03463v2#S2',
                  coefficients=[dict(term=term, powers=power, value_pb=float(c),
                                     error_pb=float(np.sqrt(v)))
                                for term, power, c, v in zip(TERMS, POWERS, coeff, np.diag(covariance))],
                  covariance_pb2=covariance.tolist(), diagnostics=summary,
                  fitted_ranges=dict(k3=[float(points[:, 0].min()), float(points[:, 0].max())],
                                     k4=[float(points[:, 1].min()), float(points[:, 1].max())]),
                  uncertainty_note='Formal covariance uses the quoted errors as absolute, assumes independent '
                  'grid points and is not rescaled by chi2/ndof. Cross-point correlations are unknown; '
                  'no theory/PDF/scale uncertainty is included. Extrapolated predictions await validation with new runs.',
                  extrapolated_to_grid=args.extrapolate)
    (args.outdir / 'fit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (args.outdir / 'residuals.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['k3', 'k4', 'xsec_pb', 'error_pb', 'fit_pb', 'data_minus_fit_pb', 'pull'])
        writer.writerows((*p, y, e, f, y - f, (y - f) / e)
                         for p, y, e, f in zip(points, values, errors, predicted))
    print('sigma [pb] = ' + ' + '.join(f'({c:.9g})*{t}' for c, t in zip(coeff, TERMS)))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    bounds = points if args.grid is None else np.loadtxt(args.grid, ndmin=2)
    k3_range = args.k3_range or [bounds[:, 0].min(), bounds[:, 0].max()]
    k4_range = args.k4_range or [bounds[:, 1].min(), bounds[:, 1].max()]
    x, y = np.meshgrid(np.linspace(*k3_range, 500), np.linspace(*k4_range, 500))
    interpolated = interpolate(points, values, x, y)
    fitted = design_matrix(x, y) @ coeff
    inside = np.isfinite(interpolated)
    if not args.extrapolate:
        fitted[~inside] = np.nan
    if args.grid is not None:
        grid_values = design_matrix(bounds[:, 0], bounds[:, 1]) @ coeff
        grid_inside = np.isfinite(interpolate(points, values, bounds[:, 0], bounds[:, 1]))
        with (args.outdir / 'grid_predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['k3', 'k4', 'fit_pb', 'extrapolated'])
            writer.writerows((k3, k4, prediction if within or args.extrapolate else '', not within)
                             for (k3, k4), prediction, within in zip(bounds, grid_values, grid_inside))
    if args.log and min(np.nanmin(fitted), np.nanmin(interpolated)) <= 0:
        parser.error('--log requires positive surface values')
    minimum = min(values.min(), np.nanmin(fitted), np.nanmin(interpolated))
    maximum = max(values.max(), np.nanmax(fitted), np.nanmax(interpolated))
    lower = minimum if args.vmin is None else args.vmin
    upper = maximum if args.vmax is None else args.vmax
    if not np.isfinite(lower + upper) or lower >= upper or (args.log and lower <= 0):
        parser.error('invalid colour scale limits')
    norm = LogNorm(lower, upper) if args.log else Normalize(lower, upper)

    def save(fig, name):
        fig.savefig(args.outdir / f'{name}.png', dpi=180)
        fig.savefig(args.outdir / f'{name}.pdf')
        plt.close(fig)

    fit_name = 'fit_extrapolated' if args.extrapolate else 'fit'
    fit_title = 'HH monomial fit — full-grid extrapolation' if args.extrapolate else 'HH monomial fit'
    for name, surface, title in [(fit_name, fitted, fit_title),
                                 ('interpolated', interpolated, 'Linear interpolation of LHE cross sections')]:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        image = ax.pcolormesh(x, y, np.ma.masked_invalid(surface), cmap='viridis', norm=norm,
                              shading='auto', rasterized=True)
        ax.scatter(*points.T, s=5, color='black', alpha=0.3, label='LHE points')
        ax.set(xlabel=r'$\kappa_3$', ylabel=r'$\kappa_4$', title=title,
               xlim=k3_range, ylim=k4_range)
        extrapolating = args.extrapolate and name == fit_name
        if extrapolating:
            # The boundary separates fitted coverage from extrapolation.
            ax.contour(x, y, inside.astype(float), levels=[0.5], colors='white', linewidths=1.3,
                       linestyles='--')
        note = ('Dashed line: edge of completed-run coverage; beyond it is extrapolation'
                if extrapolating else 'Blank: outside completed-point convex hull')
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.legend(loc='upper right')
        below = np.nanmin(surface) < lower
        above = np.nanmax(surface) > upper
        extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        fig.colorbar(image, ax=ax, label=r'$\sigma(gg\to HH)$ [pb]', extend=extend)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    percent = np.divide(100 * (values - predicted), values,
                        out=np.full_like(values, np.nan), where=values != 0)
    maximum = max(np.nanmax(np.abs(percent)), 1e-10)
    image = ax.scatter(*points.T, c=percent, cmap='coolwarm', vmin=-maximum, vmax=maximum, marker='s', s=90)
    ax.set(xlabel=r'$\kappa_3$', ylabel=r'$\kappa_4$', title='HH monomial fit residuals')
    fig.colorbar(image, ax=ax, label=r'$(\sigma_{LHE}-\sigma_{fit})/\sigma_{LHE}$ [%]')
    save(fig, 'residuals')
    print(f'Saved fit.json, CSV predictions/residuals, and {fit_name}/interpolated/residuals PNG + PDF in {args.outdir}')


if __name__ == '__main__':
    main()
