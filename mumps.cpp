
#include "dmumps_c.h"
#include "smumps_c.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <iostream>
#include <mpi.h>
#include <type_traits>

using namespace dolfinx;

template <typename T>
int mumps_solver(MPI_Comm comm, const la::MatrixCSR<T>& Amat,
                 const la::Vector<T>& bvec, la::Vector<T>& uvec, bool verbose)
{
  int rank = dolfinx::MPI::rank(comm);
  int size = dolfinx::MPI::size(comm);
  int fcomm = MPI_Comm_c2f(comm);

  typedef typename std::conditional_t<
      std::is_same_v<T, double>, DMUMPS_STRUC_C,
      std::conditional_t<std::is_same_v<T, float>, SMUMPS_STRUC_C,
                         std::false_type>>
      MUMPS_STRUC_C;

  auto mumps_c = [](MUMPS_STRUC_C* id)
  {
    if constexpr (std::is_same_v<T, double>)
      dmumps_c(id);
    else if constexpr (std::is_same_v<T, float>)
      smumps_c(id);
  };

  MUMPS_STRUC_C id;

  // Initialize
  id.job = -1;
  id.comm_fortran = fcomm;
  id.par = 1;
  id.sym = 0; // General matrix

  mumps_c(&id);

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
  int local_row_offset = Amat.index_map(0)->local_range()[0] + 1;
  std::vector<int> remote_ranges(size);
  MPI_Allgather(&local_row_offset, 1, MPI_INT, remote_ranges.data(), 1, MPI_INT,
                comm);

  for (int i = 0; i < m_loc; ++i)
  {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
      irn.push_back(i + local_row_offset);
  }

  // Convert local to global indices for columns
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), std::next(Amat.cols().begin(), nnz_loc),
                 jcn.begin(),
                 [&](std::int32_t local_index)
                 { return global_col_indices[local_index] + 1; });
  auto Amatdata = const_cast<T*>(Amat.values().data());

  assert(irn.size() == jcn.size());

  id.irn_loc = irn.data();
  id.jcn_loc = jcn.data();
  id.a_loc = Amatdata;

  id.icntl[19] = 10; // Dense RHS, distributed
  id.icntl[20] = 1;  // Distributed solution

  id.rhs_loc = const_cast<T*>(bvec.array().data());
  id.nloc_rhs = m_loc;
  id.lrhs_loc = m_loc;
  std::vector<int> irhs_loc(m_loc);
  std::iota(irhs_loc.begin(), irhs_loc.end(), local_row_offset);
  id.irhs_loc = irhs_loc.data();

  // Analyse
  id.job = 1;
  mumps_c(&id);

  // Factorize
  id.job = 2;
  mumps_c(&id);

  int lsol_loc = id.info[22];
  std::vector<T> sol_loc(lsol_loc);
  std::vector<int> isol_loc(lsol_loc);
  id.sol_loc = sol_loc.data();
  id.lsol_loc = lsol_loc;
  id.isol_loc = isol_loc.data();

  // Solve
  id.job = 3;
  mumps_c(&id);

  // Solution is permuted across processes: reorder
  std::vector<int> perm(lsol_loc);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&](int a, int b) { return isol_loc[a] < isol_loc[b]; });

  std::vector<int> isol_sort(lsol_loc);
  std::vector<T> sol_sort(lsol_loc);
  for (int i = 0; i < lsol_loc; ++i)
  {
    isol_sort[i] = isol_loc[perm[i]];
    sol_sort[i] = sol_loc[perm[i]];
  }

  // Find processor splits in data, and send to correct process
  std::vector<int> send_offsets;
  for (int i = 0; i < size; ++i)
  {
    auto it = std::lower_bound(isol_sort.begin(), isol_sort.end(),
                               remote_ranges[i]);
    send_offsets.push_back(std::distance(isol_sort.begin(), it));
  }
  send_offsets.push_back(isol_sort.size());
  std::vector<int> send_sizes(size);
  for (int i = 0; i < size; ++i)
    send_sizes[i] = send_offsets[i + 1] - send_offsets[i];

  std::vector<int> recv_sizes(size);
  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT,
               comm);
  std::vector<int> recv_offsets(size + 1, 0);
  for (int i = 0; i < size; ++i)
    recv_offsets[i + 1] = recv_offsets[i] + recv_sizes[i];

  // Send indices and data to the owning processes
  std::vector<int> recv_indices(recv_offsets.back());
  std::vector<T> recv_data(recv_offsets.back());
  MPI_Alltoallv(isol_sort.data(), send_sizes.data(), send_offsets.data(),
                MPI_INT, recv_indices.data(), recv_sizes.data(),
                recv_offsets.data(), MPI_INT, comm);
  MPI_Alltoallv(sol_sort.data(), send_sizes.data(), send_offsets.data(),
                dolfinx::MPI::mpi_type<T>(), recv_data.data(),
                recv_sizes.data(), recv_offsets.data(),
                dolfinx::MPI::mpi_type<T>(), comm);

  // Should receive exactly enough data for local part of vector
  assert(recv_data.size() == m_loc);

  // Refill into u
  auto u = uvec.mutable_array();
  for (int i = 0; i < m_loc; ++i)
    u[recv_indices[i] - local_row_offset] = recv_data[i];

  // Finalize
  id.job = -2;
  mumps_c(&id);

  uvec.scatter_fwd();
  return 0;
}

template int mumps_solver(MPI_Comm, const la::MatrixCSR<double>&,
                          const la::Vector<double>&, la::Vector<double>&, bool);

// template int mumps_solver(MPI_Comm, const la::MatrixCSR<float>&,
//                          const la::Vector<float>&, la::Vector<float>&, bool);
