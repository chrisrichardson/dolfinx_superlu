
#pragma once

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

template <typename T>
int mumps_solver(MPI_Comm comm, const la::MatrixCSR<T>& Amat,
                 const la::Vector<T>& bvec, la::Vector<T>& uvec, bool verbose=false);
