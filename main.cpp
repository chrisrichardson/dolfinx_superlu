// Poisson equation (C++)
// ======================
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Create and apply Dirichlet boundary conditions
// * Define Expressions
// * Define a FunctionSpace
//
// The solution for :math:`u` in this demo will look as follows:
//
// .. image:: ../poisson_u.png
//     :scale: 75 %
//
//
// Equation and problem definition
// -------------------------------
//
// The Poisson equation is the canonical elliptic partial differential
// equation.  For a domain :math:`\Omega \subset \mathbb{R}^n` with
// boundary :math:`\partial \Omega = \Gamma_{D} \cup \Gamma_{N}`, the
// Poisson equation with particular boundary conditions reads:
//
// .. math::
//    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
//      u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
//      \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
//
// Here, :math:`f` and :math:`g` are input data and :math:`n` denotes the
// outward directed boundary normal. The most standard variational form
// of Poisson equation reads: find :math:`u \in V` such that
//
// .. math::
//    a(u, v) = L(v) \quad \forall \ v \in V,
//
// where :math:`V` is a suitable function space and
//
// .. math::
//    a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
//    L(v)    &= \int_{\Omega} f v \, {\rm d} x
//    + \int_{\Gamma_{N}} g v \, {\rm d} s.
//
// The expression :math:`a(u, v)` is the bilinear form and :math:`L(v)`
// is the linear form. It is assumed that all functions in :math:`V`
// satisfy the Dirichlet boundary conditions (:math:`u = 0 \ {\rm on} \
// \Gamma_{D}`).
//
// In this demo, we shall consider the following definitions of the input
// functions, the domain, and the boundaries:
//
// * :math:`\Omega = [0,1] \times [0,1]` (a unit square)
// * :math:`\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}`
// (Dirichlet boundary)
// * :math:`\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}`
// (Neumann boundary)
// * :math:`g = \sin(5x)` (normal derivative)
// * :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source term)
//
//
// Implementation
// --------------
//
// The implementation is split in two files: a form file containing the
// definition of the variational forms expressed in UFL and a C++ file
// containing the actual solver.
//
// Running this demo requires the files: :download:`main.cpp`,
// :download:`poisson.py` and :download:`CMakeLists.txt`.
//
//
// UFL form file
// ^^^^^^^^^^^^^
//
// The UFL file is implemented in :download:`poisson.py`, and the
// explanation of the UFL file can be found at :doc:`here <poisson.py>`.
//
//
// C++ program
// ^^^^^^^^^^^
//
// The main solver is implemented in the :download:`main.cpp` file.
//
// At the top we include the DOLFINx header file and the generated header
// file "Poisson.h" containing the variational forms for the Poisson
// equation.  For convenience we also include the DOLFINx namespace.
//
// .. code-block:: cpp

#include "poisson.h"
#include "superlu.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = float;
using U = double;

// Then follows the definition of the coefficient functions (for
// :math:`f` and :math:`g`), which are derived from the
// :cpp:class:`Expression` class in DOLFINx
//
// .. code-block:: cpp

// Inside the ``main`` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the :cpp:class:`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified in
// the form file) defined relative to this mesh, we do as follows
//
// .. code-block:: cpp

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {64, 32}, mesh::CellType::triangle, part));

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Next, we define the variational formulation by initializing the
    // bilinear and linear forms (:math:`a`, :math:`L`) using the previously
    // defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
    // source and boundary flux term (:math:`f`, :math:`g`) and attach these
    // to the linear form.
    //
    // .. code-block:: cpp

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T, U>>(V);
    auto g = std::make_shared<fem::Function<T, U>>(V);

    // Define variational forms
    auto a = std::make_shared<fem::Form<T, U>>(fem::create_form<T, U>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T, U>>(fem::create_form<T, U>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

    // Now, the Dirichlet boundary condition (:math:`u = 0`) can be created
    // using the class :cpp:class:`DirichletBC`. A :cpp:class:`DirichletBC`
    // takes two arguments: the value of the boundary condition,
    // and the part of the boundary on which the condition applies.
    // In our example, the value of the boundary condition (0.0) can
    // represented using a :cpp:class:`Function`, and the Dirichlet boundary
    // is defined by the indices of degrees of freedom to which the boundary
    // condition applies.
    // The definition of the Dirichlet boundary condition then looks
    // as follows:
    //
    // .. code-block:: cpp

    // Define boundary condition

    auto facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
              marker[p] = true;
          }
          return marker;
        });
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T, U>>(0.0, bdofs, V);

    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(std::sin(5 * x(0, p)));
          return {f, {f.size()}};
        });

    // Now, we have specified the variational forms and can consider the
    // solution of the variational problem. First, we need to define a
    // :cpp:class:`Function` ``u`` to store the solution. (Upon
    // initialization, it is simply set to the zero function.) Next, we can
    // call the ``solve`` function with the arguments ``a == L``, ``u`` and
    // ``bc`` as follows:
    //
    // .. code-block:: cpp

    // Compute solution
    fem::Function<T, U> u(V);
    la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
    sp.finalize();
    la::MatrixCSR<T> A(sp);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    A.set(0.0);
    fem::assemble_matrix(A.mat_add_values(), *a, {bc});
    A.finalize();
    fem::set_diagonal<T, U>(A.mat_set_values(), *V, {bc});

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, U>(b.mutable_array(), {bc});

    // Solver: A.u = b
    dolfinx::common::Timer tsolve("#### SUPERLU Solver ####");
    superlu_solver(mesh->comm(), A, b, *u.x());
    tsolve.stop();

    // The function ``u`` will be modified during the call to solve. A
    // :cpp:class:`Function` can be saved to a file. Here, we output the
    // solution to a ``VTK`` file (specified using the suffix ``.pvd``) for
    // visualisation in an external program such as Paraview.
    //
    // .. code-block:: cpp

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    if constexpr (std::is_same_v<T, float>)
    {
      fem::Function<double, U> udouble(V);
      // Convert float to double
      std::copy(u.x()->array().begin(), u.x()->array().end(),
                udouble.x()->mutable_array().begin());
      file.write<double>({udouble}, 0.0);
    }
    else
      file.write<T>({u}, 0.0);
  }

  dolfinx::list_timings(MPI_COMM_WORLD, {TimingType::wall});

  MPI_Finalize();
  return 0;
}
