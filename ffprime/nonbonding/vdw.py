from atomdb import Element
import numpy as np


class TSParameters:
    def __init__(self, atnums, atvolumes, ref_atvolumes):
        self.atnums = atnums
        self.atvolumes = atvolumes
        self.ref_atvolumes = ref_atvolumes
        self.c6s_eff, self.a_eff, self.volume_ratios = TSParameters.ts_parameters(
            atnums, atvolumes, ref_atvolumes
        )

    @staticmethod
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
        c6s_eff = np.array(
            [(volume_ratios[i]) ** 2 * Element(atnum).c6["chu"] for i, atnum in enumerate(atnums)]
        )
        a_eff = np.array(
            [(volume_ratios[i]) * Element(atnum).pold["chu"] for i, atnum in enumerate(atnums)]
        )

        return c6s_eff, a_eff, volume_ratios


class XDMParameters:
    def __init__(self, dens, atvolumes, ref_atvolumes):
        pass

    @staticmethod
    def xdm_parameters(dens, atvolumes, ref_atvolumes):
        """Compute XDM dispersion parameters for a set of atoms.
        ..math::"""
        return print("not implemented yet")

    def compute_hole_parameters(self, density, gradient, laplacian, posdef_kin):
        """Solve for the exchange-hole parameters a,b for each atom."""

        def xhole_functions(x, rhs):
            """Nonlinear equations for the exchange-hole parameters a,b."""
            f = (x * np.exp((-2.0 * x / 3.0)) / (x - 2.0)) - rhs
            df = ((2.0 / 3.0) * (-(x**2.0) + 2.0 * x - 3.0) * np.exp(-2.0 * x / 3.0)) / (
                (x - 2.0) ** 2.0
            )
            return f, df

        def newton_raphson_solve(rhs, dens, max_iter=100, tol=1e-10):
            """Solve for the exchange-hole parameters a,b for each atom."""
            x0, step = 2.0, 1.0

            # Intial guess
            if rhs != 0.0:
                direction = np.sign(rhs)
                for _ in range(16):
                    x = x0 + (direction * step)
                    f = xhole_functions(x, rhs)[0]
                    if direction * f > 0.0:
                        # print(
                        #     f"Initial guess found: x0={x0}, step={step}, rhs={rhs}, f={f} in {_} steps"
                        # )
                        break
                    step *= 0.1
                else:
                    raise Exception("Failed to find initial guess for Newton-Raphson iteration.")
            else:
                print(f"rhs=0.0, returning x0={x0}")
                x = x0

            # Newton-Raphson iteration
            for _ in range(max_iter):
                f, df = xhole_functions(x, rhs)
                x1 = x - f / df
                if abs(df) < 1e-15:
                    raise Exception(
                        "Derivative of non-linear function f is too small, cannot continue Newton-Raphson iteration."
                    )
                if abs(x1 - x) < tol:
                    # print(f"Converged after {_} iterations: x={x1}, rhs={rhs}, f={f}")
                    break
                x = x1

            # Compute a and b from x
            a = ((8 * np.pi * dens) / np.exp(-x)) ** (1.0 / 3.0)
            b = x / a
            # print(f"Returning computed b={b}")
            return a, b

        a = np.zeros_like(density, dtype=float)
        b = np.zeros_like(density, dtype=float)

        for i, dens in enumerate(density):
            dens = np.clip(dens, 1e-10, None)  # Avoid division by zero or negative densities
            grad = gradient[i]
            lap = laplacian[i]
            tau = posdef_kin[i]
            Q_sigma = (1.0 / 6.0) * (lap - (2.0 * tau) + ((1.0 / 2.0) * np.sum(grad**2) / dens))
            Q_sigma = np.clip(Q_sigma, 1e-10, None)  # Avoid division by zero or negative Q_sigma
            rhs = (2.0 / 3.0) * (np.pi * dens) ** (2.0 / 3.0) * dens / Q_sigma

            a[i], b[i] = newton_raphson_solve(rhs, dens)

        return a, b

    def exchange_hole(self):
        """Compute the exchange-hole for each atom."""
        return print("not implemented yet")

    def exchange_hole_moments(self):
        """Compute the exchange-hole multipole moments for each atom."""
        return print("not implemented yet")
