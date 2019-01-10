import os
import numpy as np
from dolfin import *
import parameters as params

def extract_pots(phi, positions):
    compt_values = np.zeros(positions.shape[0])
    for ii in range(positions.shape[0]):
        compt_values[ii] = phi(positions[ii, :])
    return compt_values

def main_4shell_fem(mesh, subdomains, boundaries, skull_cond, src_pos, snk_pos):
    sigma_B = Constant(params.sigma_brain)
    sigma_Sc = Constant(params.sigma_scalp)
    sigma_C = Constant(params.sigma_csf)
    sigma_Sk = Constant(skull_cond)
    sigma_A = Constant(0.)

    V = FunctionSpace(mesh, "CG", 2)
    v = TestFunction(V)
    u = TrialFunction(V)

    phi = Function(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)
    a = inner(sigma_B * grad(u), grad(v))*dx(params.whitemattervol) + \
        inner(sigma_B * grad(u), grad(v))*dx(params.graymattervol) + \
        inner(sigma_Sc * grad(u), grad(v))*dx(params.scalpvol) + \
        inner(sigma_C * grad(u), grad(v))*dx(params.csfvol) + \
        inner(sigma_Sk * grad(u), grad(v))*dx(params.skullvol)
    L = Constant(0)*v*dx
    A = assemble(a)
    b = assemble(L)

    x_pos, y_pos, z_pos = src_pos
    point = Point(x_pos, y_pos, z_pos)
    delta = PointSource(V, point, 1.)
    delta.apply(b)

    x_pos, y_pos, z_pos = snk_pos
    point1 = Point(x_pos, y_pos, z_pos)
    delta1 = PointSource(V, point1, -1.)
    delta1.apply(b)

    solver = KrylovSolver("cg", "ilu")
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["absolute_tolerance"] = 1E-8
    solver.parameters["monitor_convergence"] = True

    info(solver.parameters, True)
#    set_log_level(PROGRESS) does not work in fenics 2018.1.0
    solver.solve(A, phi.vector(), b)

    ele_pos_list = params.ele_coords
    vals = extract_pots(phi, ele_pos_list)
    # np.save(os.path.join('results', save_as), vals)
    return vals

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', nargs='?',
                        default=os.path.join('mesh', 'sphere_4.xml'),
                        help='a path to the XML file with the mesh')
    parser.add_argument('--directory', '-d',
                        default='results',
                        dest='results',
                        help='a path to the result directory')

    args = parser.parse_args()
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    print('Loading meshes')
    mesh = Mesh(args.mesh)
    subdomains = MeshFunction("size_t", mesh, args.mesh[:-4] + '_physical_region.xml')
    boundaries = MeshFunction("size_t", mesh, args.mesh[:-4] + '_facet_region.xml')
    for dipole in params.dipole_list:
        print('Now computing FEM for dipole: ', dipole['name'])
        src_pos = dipole['src_pos']
        snk_pos = dipole['snk_pos']
        print('Done loading meshes')
        fem_20 = main_4shell_fem(mesh, subdomains, boundaries,
                                 params.sigma_skull20, src_pos, snk_pos)
        print('Done 4Shell-FEM-20')
        fem_40 = main_4shell_fem(mesh, subdomains, boundaries,
                                 params.sigma_skull40, src_pos, snk_pos)
        print('Done 4Shell-FEM-40')
        fem_80 = main_4shell_fem(mesh, subdomains, boundaries,
                                 params.sigma_skull80, src_pos, snk_pos)
        print('Done 4Shell-FEM-80')
        f = open(os.path.join(args.results,
                              'Numerical_' + dipole['name'] + '.npz'), 'wb')
        np.savez(f, fem_20=fem_20, fem_40=fem_40, fem_80=fem_80)
        f.close()
