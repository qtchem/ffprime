"""
Shared helpers for the electrostatics package.

``cartesian.py`` and ``spherical.py`` both need to (a) validate the shapes
of the coordinate/point arrays they're given, and (b) compute pairwise
displacement vectors and singularity-safe distances between multipole
sites and field points. Both pieces of logic were previously duplicated
verbatim in each module; they now live here as the single source of
truth.

Nothing in this module performs a physics calculation on its own -- it
only prepares/validates the inputs that ``cartesian.py`` and
``spherical.py`` build their formulas on top of.
"""

import numpy as np
from denspart.properties import compute_multipole_moments


def validate_shapes(coords, points):
    """Validate coordinate and point array shapes.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Multipole-site coordinates.
    points : np.ndarray, shape (M, 3)
        Field points at which a potential/field will be evaluated.

    Raises
    ------
    ValueError
        If ``coords`` or ``points`` does not have shape ``(*, 3)``.
    """
    coords = np.asarray(coords)
    points = np.asarray(points)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (M, 3), got {points.shape}")


def compute_displacement(coords, points):
    """Compute pairwise displacement vectors and singularity-safe distances.

    This is the ``r_vecs`` / ``r`` / ``safe_r`` block that every potential
    and field routine in this package starts with:

        r_vecs = points[:, None, :] - coords[None, :, :]
        r      = ||r_vecs||
        safe_r = r, with any distance below 1e-12 replaced by inf

    Replacing near-zero distances with ``inf`` keeps evaluation at (or
    extremely close to) a multipole's own site finite, rather than
    raising a division-by-zero warning.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    r_vecs : np.ndarray, shape (M, N, 3)
        Displacement vectors ``points[m] - coords[n]``.
    r : np.ndarray, shape (M, N)
        Euclidean distances corresponding to ``r_vecs``.
    safe_r : np.ndarray, shape (M, N)-
        ``r`` with near-zero entries replaced by ``inf``.
    """
    coords = np.asarray(coords)
    points = np.asarray(points)

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)

    return r_vecs, r, safe_r


def compute_cartesian_atomic_moments(pro_model, grid, moldens, localgrids):
    """Compute per-atom Cartesian charge/dipole/quadrupole moments.

    ``compute_multipole_moments`` returns pure (real) spherical
    moments per atom, ``pm``, ordered as:

        pm[0:3] -> l=1 moments (Stone order: Q_10, Q_11c, Q_11s)
        pm[3:8] -> l=2 moments (Q_20, Q_21c, Q_21s, Q_22c, Q_22s)

    ``pm`` is expressed in terms of the *electron* density, whereas
    Stone's convention (and the physical, nuclear-frame multipole we
    want to expose) is defined in terms of the *charge* density
    rho_c = -rho_e. That is the only discrepancy between ``pm`` and a
    genuine Stone spherical multipole: there is no extra sqrt(3)/2
    rescaling needed on the m=+-1/+-2 components, because
    ``compute_multipole_moments`` already returns them with the correct
    Racah normalization expected by
    ``ffprime.electrostatics.spherical.quadrupole_spherical_to_cartesian``.

    We therefore only flip the overall sign (electron density -> charge
    density) before delegating the actual spherical -> Cartesian
    conversion to the shared, tested utilities
    ``dipole_spherical_to_cartesian`` and
    ``quadrupole_spherical_to_cartesian``.
    """

    from ffprime.electrostatics.spherical import (
        dipole_spherical_to_cartesian,
        quadrupole_spherical_to_cartesian,
    )

    cartesian_moments = []
    pure_moments = compute_multipole_moments(pro_model, grid, moldens, localgrids)

    for i, pm in enumerate(pure_moments):
        # atomic charge
        atmom_cart = [pro_model.charges[i]]

        # ---- Dipole ----
        # pm[0:3] are already in Stone order [Q_10, Q_11c, Q_11s].
        # Only the electron-density -> charge-density sign flip is
        # needed before converting to Cartesian form.
        dipole_cart = -dipole_spherical_to_cartesian(pm[0:3])
        atmom_cart.append(dipole_cart[0])  # x
        atmom_cart.append(dipole_cart[1])  # y
        atmom_cart.append(dipole_cart[2])  # z

        # ---- Quadrupole ----
        # pm[3:8] are already genuine Stone spherical quadrupole
        # components (Racah-normalized); only the electron-density ->
        # charge-density sign flip is applied.
        Q20 = -pm[3]
        Q21c = -pm[4]
        Q21s = -pm[5]
        Q22c = -pm[6]
        Q22s = -pm[7]
        quad_sph = np.array([Q20, Q21c, Q21s, Q22c, Q22s])

        theta = quadrupole_spherical_to_cartesian(quad_sph)

        # Preserve original flattened ordering: xx, xy, xz, yy, yz, zz
        atmom_cart.append(theta[0, 0])  # xx
        atmom_cart.append(theta[0, 1])  # xy
        atmom_cart.append(theta[0, 2])  # xz
        atmom_cart.append(theta[1, 1])  # yy
        atmom_cart.append(theta[1, 2])  # yz
        atmom_cart.append(theta[2, 2])  # zz

        cartesian_moments.append(atmom_cart)

    return np.array(cartesian_moments)
