import numpy as np
from atomdb import load
from scipy.integrate import quad
from denspart.properties import compute_radial_moments


def compute_volumes(atnums, unique_atnums, pro_model, grid, moldens, localgrids):
    ref_volumes = np.zeros(len(atnums))
    volumes = np.zeros(len(atnums))
    volume_ratios = np.zeros(len(atnums))

    # if proatomdb is not None:
    ref_vol = {}
    elem2mult = {1: 2, 6: 3, 7: 4, 8: 3, 16: 6, 15: 4}
    for i in unique_atnums:
        proat_db = load(elem=i, charge=0, mult=elem2mult[i], dataset="gaussian")
        dens_spline = proat_db.dens_func(spin="t", log=True)
        r_max = 1000.0  # or just 6.0
        integrand = lambda r: 4 * np.pi * (r**5) * dens_spline(np.array([r]))[0]
        rcubic, err = quad(integrand, 0, r_max)
        ref_vol[i] = rcubic

    radial_moments = compute_radial_moments(pro_model, grid, moldens, localgrids)[:, 3]

    for i, atnum in enumerate(atnums):
        # store atomic volume as the 3rd atomic radial moment
        volumes[i] = radial_moments[i]
        # store reference atomic volume as the 3rd radial moment of neutral atom
        ref_volumes[i] = ref_vol[atnum]

    return ref_volumes, volumes
