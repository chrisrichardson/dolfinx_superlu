
#include "dmumps_c.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <iostream>
#include <mpi.h>

using namespace dolfinx;

using T = double;

template <typename T>
int mumps_solver(MPI_Comm comm, const la::MatrixCSR<T>& Amat,
                 const la::Vector<T>& bvec, la::Vector<T>& uvec, bool verbose)
{
  int rank = dolfinx::MPI::rank(comm);
  int size = dolfinx::MPI::size(comm);
  int fcomm = MPI_Comm_c2f(comm);

  DMUMPS_STRUC_C id;

  // Initialize
  id.job = -1;
  id.comm_fortran = fcomm;
  id.par = 1;
  id.sym = 0; // General matrix
  dmumps_c(&id);

  id.icntl[4] = 0;  // Assembled matrix
  id.icntl[17] = 3; // Fully distributed

  // Global size
  int m = Amat.index_map(0)->size_global();
  int n = Amat.index_map(1)->size_global();
  if (m != n)
    throw std::runtime_error("Can't solve non-square system");
  id.n = n;

  // Number of local rows
  int m_loc = Amat.num_owned_rows();

  // Local number of non-zeros
  auto row_ptr = Amat.row_ptr();
  int nnz_loc = row_ptr[m_loc];
  id.nnz_loc = nnz_loc;

  // Row and column indices (+1 for FORTRAN style)
  std::vector<int> irn;
  irn.reserve(nnz_loc);
  std::vector<int> jcn(nnz_loc);

  // Create row indices
  int row = Amat.index_map(0)->local_range()[0];
  std::vector<int> remote_ranges(size);
  MPI_Allgather(&row, 1, MPI_INT, remote_ranges.data(), 1, MPI_INT, comm);

  for (int i = 0; i < m_loc; ++i)
  {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
      irn.push_back(row + 1);
    ++row;
  }

  // Convert local to global indices for columns
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), std::next(Amat.cols().begin(), nnz_loc),
                 jcn.begin(),
                 [&](std::int32_t local_index)
                 { return global_col_indices[local_index] + 1; });
  auto Amatdata = const_cast<T*>(Amat.values().data());

  std::cout << "irn = " << irn.size() << "\n";
  std::cout << "jcn = " << jcn.size() << "\n";
  std::cout << "nnz_loc = " << nnz_loc << "\n";
  std::cout << "m_loc = " << m_loc << "\n";
  std::cout << "m = " << m << "\n";

  assert(irn.size() == jcn.size());

  id.irn_loc = irn.data();
  id.jcn_loc = jcn.data();
  id.a_loc = Amatdata;

  id.icntl[19] = 10; // Dense RHS, distributed
  id.icntl[20] = 1;  // Distributed solution

  std::vector<T> rhs;
  if (rank == 0)
  {
    rhs.resize(m, 0.0);
    id.rhs = rhs.data();
  }

  id.rhs_loc = const_cast<T*>(bvec.array().data());
  id.nloc_rhs = m_loc;
  id.lrhs_loc = m_loc;
  std::vector<int> irhs_loc(m_loc);
  id.irhs_loc = irhs_loc.data();

  // Analyse
  id.job = 1;
  dmumps_c(&id);

  // Factorize
  id.job = 2;
  dmumps_c(&id);

  int lsol_loc = id.info[22];
  std::vector<T> sol_loc(lsol_loc);
  std::vector<int> isol_loc(lsol_loc);
  id.sol_loc = sol_loc.data();
  id.lsol_loc = lsol_loc;
  id.isol_loc = isol_loc.data();

  // Solve
  id.job = 3;
  dmumps_c(&id);

  std::vector<int> isol_sort(isol_loc.begin(), isol_loc.end());
  std::sort(isol_sort.begin(), isol_sort.end());
  // Find processor splits in data
  std::stringstream s;
  for (int i = 0; i < size; ++i)
  {
    auto it = std::lower_bound(isol_sort.begin(), isol_sort.end(),
                               remote_ranges[i]);
    s << std::distance(isol_sort.begin(), it) << " ";
  }
  s << "\n";

  s << "lsol_loc = " << lsol_loc << " " << m_loc << "\n";
  for (auto q : isol_sort)
    s << q << ",";
  std::cout << s.str() << std::endl;

  // Finalize
  id.job = -2;
  dmumps_c(&id);
  return 0;
}

template int mumps_solver(MPI_Comm, const la::MatrixCSR<double>&,
                          const la::Vector<double>&, la::Vector<double>&, bool);
