from atomdb import Element


def ts_parameters(atnums, atvolumes, ref_atvolumes):
    """Compute Tkatchenko-Scheffler dispersion parameters for a set of atoms.
    ..math::
        C6_eff = (V_eff / V_ref)^2 * C6_free
        a_eff = (V_eff / V_ref) * a_free

    Parameters
    ----------
    atnums : array_like of int
        Atomic numbers of the atoms, shaped ``(n_atoms,)``.
    atvolumes : array_like of float
        Effective atomic volumes in the molecular environment, shaped ``(n_atoms,)``.
    ref_atvolumes : array_like of float
        Reference free-atom volumes, shaped ``(n_atoms,)``.

    Returns
    -------
    c6s_eff : np.ndarray
        Environment-dependent atomic C6 dispersion coefficients,
        shaped ``(n_atoms,)``.
    a_eff : np.ndarray
        Environment-dependent atomic static polarizabilities,
        shaped ``(n_atoms,)``.
    volume_ratios : np.ndarray
        Ratios of effective to reference atomic volumes,
        shaped ``(n_atoms,)``.

    Notes
    -----
    The Tkatchenko-Scheffler method rescales free-atom properties
    according to the ratio of effective to reference atomic volumes.
    The C6 coefficients scale quadratically with the volume ratio,
    while static polarizabilities scale linearly.
    """
    volume_ratios = np.array([atvolumes[i] / ref_atvolumes[i] for i in range(len(atnums))])
    c6s_eff = np.array([(volume_ratios[i])**2 * Element(atnum).c6["chu"] for i, atnum in enumerate(atnums)])
    a_eff = np.array([(volume_ratios[i]) * Element(atnum).pold["chu"] for i, atnum in enumerate(atnums)])

    return c6s_eff, a_eff, volume_ratios

def xdm_parameters(dens, atvolumes, ref_atvolumes):
    """Compute XDM dispersion parameters for a set of atoms.
    ..math::"""
    return print("not implemented yet")
    