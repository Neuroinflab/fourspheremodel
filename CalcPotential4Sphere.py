# -*- coding: utf-8 -*-
from __future__ import division
from scipy.special import eval_legendre, lpmv
import matplotlib.pyplot as plt
import numpy as np
# from plotting.PlotLFP import PlotLFP


class CalcPotential4Sphere:
    """ Import all data needed for calculating the extracellular potential Phi at location r
    from a current dipole moment p at location rz. Also decompose p into a radial and a tangential part.
    """

    def __init__(self, radii, sigmas, r, rz):

        self.k1 = 1. #1E6  # from mV to nV
        self.r1 = radii[0]
        self.r2 = radii[1]
        self.r3 = radii[2]
        self.r4 = radii[3]

        self.sigma1 = sigmas[0]
        self.sigma2 = sigmas[1]
        self.sigma3 = sigmas[2]
        self.sigma4 = sigmas[3]

        self.r12 = self.r1 / self.r2
        self.r21 = self.r2 / self.r1
        self.r23 = self.r2 / self.r3
        self.r32 = self.r3 / self.r2
        self.r34 = self.r3 / self.r4
        self.r43 = self.r4 / self.r3

        self.sigma12 = self.sigma1 / self.sigma2
        self.sigma21 = self.sigma2 / self.sigma1
        self.sigma23 = self.sigma2 / self.sigma3
        self.sigma32 = self.sigma3 / self.sigma2
        self.sigma34 = self.sigma3 / self.sigma4
        self.sigma43 = self.sigma4 / self.sigma3

        self.rxyz = r
        self.rzloc = rz
        self.rz = np.sqrt(np.sum(rz ** 2))
        self.rz1 = self.rz / self.r1
        self.r = np.sqrt(np.sum(r ** 2, axis=1))

    def calc_potential(self, p):
        """
        Parameters
        __________
        p : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns
        _______
        potential : ndarray [nV]
            Array containing the electric potential at point self.r.

        """
        # print 'calc potential'
        print('p', p)
        p_rad, p_tan = self.decompose_dipole(p)
        print('p_rad', p_rad)
        print('p_tan', p_tan)
        pot_rad = self.calc_rad_potential(p_rad)
        pot_tan = self.calc_tan_potential(p_tan)

        pot_tot = pot_rad + pot_tan
        return pot_tot

    def decompose_dipole(self, p):
        """Decompose current dipole moment vector in radial and tangential terms
        Parameters
        __________
        p : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns:
        ________
        p_rad : ndarray [1E-15 Am]
                Radial part of p, parallel to self.rz
        p_tan : ndarray [1E-15 Am]
                Tangential part of p, orthogonal to self.rz
        """
        p_rad = (np.dot(p, self.rzloc)/self.rz ** 2).reshape(len(p),1) * self.rzloc.reshape(1, len(self.rzloc))
        p_tan = p - p_rad

        return p_rad, p_tan

    def calc_rad_potential(self, p_rad):
        """Return potential from radial dipole p_rad at location rz measured at r
        Parameters
        __________
        P : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns
        _______
        potential : ndarray [nV]
            Array containing the current dipole moment at point r.
        """

        p_tot = np.linalg.norm(p_rad, axis=1)
        print(p_tot)
        theta = self.calc_theta()
        s_vector = self.sign_rad_dipole(p_rad)
        phi_const = s_vector * p_tot / (4 * np.pi * self.sigma1 * self.rz ** 2) * self.k1
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            el_rad = self.r[el_point]
            theta_point = theta[el_point]
            if el_rad <= self.r1:
                n_terms[el_point] = self.potential_brain_rad(el_rad, theta_point)
            elif el_rad <= self.r2:
                n_terms[el_point] = self.potential_csf_rad(el_rad, theta_point)
            elif el_rad <= self.r3:
                n_terms[el_point] = self.potential_skull_rad(el_rad, theta_point)
            elif el_rad <= (self.r4):
                n_terms[el_point] = self.potential_scalp_rad(el_rad, theta_point)
            elif el_rad <= (self.r4+1E-6):
                el_rad = self.r4
                n_terms[el_point] = self.potential_scalp_rad(el_rad, theta_point)
            else:
                n_terms[el_point] = np.nan
                raise ValueError('Electrode located outside head model. Maximum r = %s µm.', self.r4, '\n your r = ', self.r)
        potential = phi_const * n_terms
        return potential

    def calc_tan_potential(self, p_tan):
        """Return potential from tangential dipole P at location rz measured at r
        Parameters
        __________
        p : ndarray [1E-15 Am]
            Array containing the current dipole moment for all timesteps in
            the x-, y- and z-direction.

        Returns
        _______
        potential : ndarray [nV]
            Array containing the current dipole moment at point r.
        """
        theta = self.calc_theta()
        phi = self.calc_phi(p_tan)
        p_tot = np.linalg.norm(p_tan, axis=1)
        phi_hom = - p_tot / (4 * np.pi * self.sigma1 * self.rz ** 2) * np.sin(phi) * self.k1
        n_terms = np.zeros((len(self.r), len(p_tot)))
        for el_point in range(len(self.r)):
            el_rad = self.r[el_point]
            theta_point = theta[el_point]
            if el_rad <= self.r1:
                n_terms[el_point] = self.potential_brain_tan(el_rad, theta_point)
            elif el_rad <= self.r2:
                n_terms[el_point] = self.potential_csf_tan(el_rad, theta_point)
            elif el_rad <= self.r3:
                n_terms[el_point] = self.potential_skull_tan(el_rad, theta_point)
            elif el_rad <= self.r4:
                n_terms[el_point] = self.potential_scalp_tan(el_rad, theta_point)
            else:
                n_terms[el_point] = np.nan
        potential = phi_hom * n_terms
        return potential

    def calc_theta(self):
        """We define z-axis in the direction of the radial part of the dipole,
        which means that the z-axis == the rz-vector."""
        cos_theta = np.dot(self.rxyz, self.rzloc) / (np.linalg.norm(self.rxyz, axis=1) * np.linalg.norm(self.rzloc))
        cos_theta = np.nan_to_num(cos_theta)
        theta = np.arccos(cos_theta)
        return theta

    def calc_phi(self, p):
        """Calculate angle phi for tangential dipole
           Parameters
           __________
           p : ndarray [1E-15 Am]
               Array containing tangential component of current dipole moment
           Returns
           _______
           phi : ndarray [radians]
                 Array containing azimuthal angle between x-axis and
                 projection of electrode location vector rxyz into xy-plane,
                 rxy.
                 Z-axis is defined in the direction of rzloc.
                 Y-axis is defined in the direction of p (orthogonal to rzloc).
                 X-axis is defined as cross product between p and rzloc (x).
                 """
        proj_rxyz_rz = (np.dot(self.rxyz,
                        self.rzloc) / np.sum(self.rzloc **
                        2)).reshape(len(self.rxyz),1) * self.rzloc.reshape(1,3)
                        # projection of rxyz onto rzloc
        rxy = self.rxyz - proj_rxyz_rz  # projection of rxyz into xy-plane
        x = np.cross(p, self.rzloc)  # vector giving direction of x-axis
        cos_phi = np.dot(rxy, x.T)/np.dot(np.linalg.norm(rxy,
                         axis=1).reshape(len(rxy),1),np.linalg.norm(x,
                         axis=1).reshape(1, len(x)))
        cos_phi = np.nan_to_num(cos_phi)
        phi_temp = np.arccos(cos_phi) # nb: phi_temp is in range [0, pi]
        phi = phi_temp
        range_test = np.dot(rxy, p.T)  # if range_test < 0, phi > pi
        for i in range(len(self.r)):
            for j in range(len(p)):
                if range_test[i,j] < 0:
                    phi[i,j] = 2*np.pi - phi_temp[i,j]
        return phi

    def sign_rad_dipole(self, p):
        """Flip radial dipole pointing inwards (i.e. we only use p_tot),
        and add a -1 to the s-vector, so that the potential can be
        calculated as if the dipole were pointing outwards,
        and later be multiplied by -1 to get the right potential."""
        sign_vector = np.ones(len(p))
        radial_test = np.dot(p, self.rzloc) / (np.linalg.norm(p, axis=1) * self.rz)
        for i in range(len(p)):
            if np.abs(radial_test[i] + 1) < 10 ** -8:
                sign_vector[i] = -1.
        return sign_vector

    def potential_brain_rad(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """

        n = np.arange(1, 100)
        c1n = self.calc_c1n(n)
        consts = n*(c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_csf_rad(self, r, theta):
        """Calculate potential in CSF from radial dipole
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c2n = self.calc_c2n(n)
        d2n = self.calc_d2n(n, c2n)
        consts = n*(c2n * (r / self.r2) ** n + d2n * (self.r2 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_skull_rad(self, r, theta):
        """Calculate potential in skull from radial dipole
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c3n = self.calc_c3n(n)
        d3n = self.calc_d3n(n, c3n)
        consts = n*(c3n * (r / self.r3) ** n + d3n * (self.r3 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_scalp_rad(self, r, theta):
        """Calculate potential in scalp from radial dipole
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c4n = self.calc_c4n(n)
        d4n = self.calc_d4n(n, c4n)
        consts = n*(c4n * (r / self.r4) ** n + d4n * (self.r4 / r) ** (n + 1))
        consts = np.insert(consts, 0, 0)
        leg_consts = np.polynomial.legendre.Legendre(consts)
        pot_sum = leg_consts(np.cos(theta))
        return pot_sum

    def potential_brain_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c1n = self.calc_c1n(n)
        consts = (c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def potential_csf_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c2n = self.calc_c2n(n)
        d2n = self.calc_d2n(n, c2n)
        consts = c2n*(r/self.r2)**n + d2n*(self.r2/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def potential_skull_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c3n = self.calc_c3n(n)
        d3n = self.calc_d3n(n, c3n)
        consts = c3n*(r/self.r3)**n + d3n*(self.r3/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def potential_scalp_tan(self, r, theta):
        """Calculate sum with constants and legendres
        Parameters
        __________
        r : ndarray [micro m]
            Array containing electrode location.
        theta : ndarray [radians]
                Array of angles between electrode location and
                dipole location vectors.
        p :     ndarray [1Ee-15 Am]
                Array containing the current dipole moment for all timesteps in
                the x-, y- and z-direction.
        Returns
        _______
        pot_sum : float
                Sum factor containing brain constants, dipole and
                electrode locations and legendre polynomials.
        """
        n = np.arange(1,100)
        c4n = self.calc_c4n(n)
        d4n = self.calc_d4n(n, c4n)
        consts = c4n*(r/self.r4)**n + d4n*(self.r4/r)**(n+1)
        pot_sum = np.sum([c*lpmv(1, i, np.cos(theta)) for c,i in zip(consts,n)])
        return pot_sum

    def calc_vn(self, n):
        r_const = (self.r34 ** n - self.r43 ** (n + 1)) / ((n + 1) / n * self.r34 ** n + self.r43 ** (n + 1))
        v = (n / (n + 1) * self.sigma34 - r_const) / (self.sigma34 + r_const)
        return v

    def calc_yn(self, n):
        vn = self.calc_vn(n)
        r_const = (n / (n + 1) * self.r23 ** n - vn * self.r32 ** (n + 1)) / (self.r23 ** n + vn * self.r32 ** (n + 1))
        y = (n / (n + 1) * self.sigma23 - r_const) / (self.sigma23 + r_const)
        return y

    def calc_zn(self, n):
        yn = self.calc_yn(n)
        z = (self.r12 ** n - (n + 1) / n * yn * self.r21 ** (n + 1)) / (self.r12 ** n + yn * self.r21 ** (n + 1))
        return z

    def calc_c1n(self, n):
        zn = self.calc_zn(n)
        c = ((n + 1) / n * self.sigma12 + zn) / (self.sigma12 - zn) * self.rz1**(n+1)
        return c

    def calc_c2n(self, n):
        yn = self.calc_yn(n)
        c1 = self.calc_c1n(n)
        c2 = (c1 + self.rz1**(n+1)) / (self.r12 ** n + yn * self.r21 ** (n + 1))
        return c2

    def calc_d2n(self, n, c2):
        yn = self.calc_yn(n)
        d2 = yn * c2
        return d2

    def calc_c3n(self, n):
        vn = self.calc_vn(n)
        c2 = self.calc_c2n(n)
        d2 = self.calc_d2n(n, c2)
        c3 = (c2 + d2) / (self.r23 ** n + vn * self.r32 ** (n + 1))
        return c3

    def calc_d3n(self, n, c3):
        vn = self.calc_vn(n)
        d3 = vn * c3
        return d3

    def calc_c4n(self, n):
        c3 = self.calc_c3n(n)
        d3 = self.calc_d3n(n, c3)
        c4 = (n + 1) / n * (c3 + d3) / ((n + 1) / n * self.r34 ** n + self.r43 ** (n + 1))
        return c4

    def calc_d4n(self, n, c4):
        d4 = n / (n + 1) * c4
        return d4


if __name__ == '__main__':
    import parameters as params
    import argparse
    import os.path

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d',
                        default='results',
                        dest='results',
                        help='a path to the result directory')

    args = parser.parse_args()

    # XXX: file results/eeg_rad200000.npz removed
    # by e5bef9051098e6790afbc434a9acc6d9a2b229e3
    # If fixed - undo 2450b755a8870220143d400149f98b66bf1fd1b8
    fle = np.load(os.path.join(args.results,
                               'eeg_rad200000.npz'))
    param_dict = fle['params'].item()

    radii = param_dict['radii']
    sigmas = [params.sigma_brain, params.sigma_csf,
              params.sigma_skull20, params.sigma_scalp]  # param_dict['sigmas']
    el_points = params.ele_coords
    charge_pos = [ii / 10000. for ii in param_dict['charge_pos']]
    charges = [-1 * ii / 200000 for ii in param_dict['charge']]
    radii = [ii / 10000.0 for ii in radii]

    print(sigmas, el_points, charge_pos, charges, radii)

    phis = []
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])

    dipole = params.dipole_list[0]
    charge_pos = [np.array(dipole['src_pos']), np.array(dipole['snk_pos'])]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    dipole = params.dipole_list[1]
    charge_pos = [np.array(dipole['src_pos']), np.array(dipole['snk_pos'])]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    dipole = params.dipole_list[2]
    charge_pos = [np.array(dipole['src_pos']), np.array(dipole['snk_pos'])]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [-0.05, 0., 7.8]  # y = 0 plane
    snk_pos = [0.05, 0., 7.8]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [0, -0.05, 7.8]  # x = 0 plane
    snk_pos = [0, 0.05, 7.8]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [0.05, 7.8, 0]  # z = 0 plane
    snk_pos = [-0.05, 7.8, 0]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [7.8, -0.05, 0]  # z = 0 plane
    snk_pos = [7.8, 0.05, 0]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [7.85, 0.0, 0.0]  # radian along x
    snk_pos = [7.75, 0.0, 0.0]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [0.0, 7.85, 0.0]  # radial along y
    snk_pos = [0.0, 7.75, 0.0]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    src_pos = [0.0, 0.0, 7.75]  # radian along z
    snk_pos = [0.0, 0.0, 7.85]
    charge_pos = [np.array(src_pos), np.array(snk_pos)]
    rz1 = (charge_pos[0] + charge_pos[1]) / 2
    P1 = np.array([charge_pos[0] * charges[0] + charge_pos[1] * charges[1]])
    sphere_mod = CalcPotential4Sphere(radii, sigmas, el_points, rz1)
    phis.append(sphere_mod.calc_potential(P1).reshape(180, 180))

    with open(os.path.join(args.results,
                           'CalcPotential4_correct_all.npz'),
              'wb') as f:
      np.savez(f, phis[0],phis[1],phis[2],phis[3],phis[4],phis[5],phis[6],phis[7],phis[8], phis[9])
