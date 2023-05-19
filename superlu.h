
#pragma once

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

/// Solve A.u = b with SuperLU_dist
/// @param comm MPI_Comm
/// @param Amat CSR Matrix, distributed by row and finalized
/// @param bvec RHS vector
/// @param uvec Solution vector
int superlu_solver(MPI_Comm comm, dolfinx::la::MatrixCSR<double>& Amat,
                   const dolfinx::la::Vector<double>& bvec,
                   dolfinx::la::Vector<double>& uvec);
