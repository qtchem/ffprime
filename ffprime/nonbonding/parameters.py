import scipy.constants as spc

M_TO_BOHR = 1.0 / spc.value("Bohr radius")
NM_TO_BOHR = spc.nano * M_TO_BOHR
KJMOL_TO_HARTREE = spc.kilo / (spc.Avogadro * spc.value("Hartree energy"))

def lj_parameters(c6s_eff, a_eff):
    """
    Compute Lennard-Jones parameters from Tkatchenko-Scheffler dispersion parameters.

    Parameters
    ----------
    c6s_eff : array_like of float
        Environment-dependent atomic C6 dispersion coefficients,
        shaped ``(n_atoms,)``.
    a_eff : array_like of float
        Environment-dependent atomic static polarizabilities,
        shaped ``(n_atoms,)``.

    Returns
    -------
    sigma : np.ndarray
        Lennard-Jones sigma parameters (in nm),
        shaped ``(n_atoms,)``.
    epsilon : np.ndarray
        Lennard-Jones epsilon parameters (in kJ/mol),
        shaped ``(n_atoms,)``.
    """
    sigma = (5.08 * a_eff**(1/7)) / (2**(1/6)) / NM_TO_BOHR
    epsilon = c6s_eff / (2 * (5.08 * a_eff**(1/7))**6) / KJMOL_TO_HARTREE

    return sigma, epsilon

