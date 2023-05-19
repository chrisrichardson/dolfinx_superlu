
#include "superlu_ddefs.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

using namespace dolfinx;

int superlu_solver(MPI_Comm comm, la::MatrixCSR<double>& Amat,
                   const la::Vector<double>& bvec, la::Vector<double>& uvec)
{
  int rank, size;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  gridinfo3d_t grid;
  superlu_gridinit3d(comm, 1, 1, size, &grid);

  printf("iam = %d\n", grid.iam);
  fflush(stdout);

  printf("size = %d\n", size);
  fflush(stdout);

  // Number of local rows
  int m_loc = Amat.num_owned_rows();
  int m;
  MPI_Allreduce(&m_loc, &m, 1, MPI_INT, MPI_SUM, comm);

  // Global size
  int n = m;

  // First row?
  int fst_row = 0;
  MPI_Scan(&m_loc, &fst_row, 1, MPI_INT, MPI_SUM, comm);
  fst_row -= m_loc;

  SuperMatrix A;

  // Copy from int64
  std::vector<int_t> row_ptr(Amat.row_ptr().begin(), Amat.row_ptr().end());

  // Convert local to global indices (and cast to int_t)
  std::vector<int_t> cols(Amat.cols().size());
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), Amat.cols().end(), cols.begin(),
                 [&](std::int64_t local_index)
                 { return global_col_indices[local_index]; });
  // Local number of non-zeros
  int nnz_loc = row_ptr[m_loc];

  std::stringstream s;

#if 0
  s << "row_ptr.size = " << row_ptr.size() << std::endl;
  s << "nnz_loca = " << nnz_loc << std::endl;
  s << "m = " << m << std::endl;
  s << "n = " << n << std::endl;
  s << "m_loc = " << m_loc << std::endl;
  s << "fst row = " << fst_row << "\n";

  for (int i = 0; i < m_loc; ++i)
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
      if (cols.data()[j] == i + fst_row)
        s << "diag (" << i + fst_row << ") = " << Amat.values().data()[j]
          << "\n";

  std::cout << s.str() << "\n";
#endif

  dCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, fst_row,
                                 Amat.values().data(), cols.data(),
                                 row_ptr.data(), SLU_NR_loc, SLU_D, SLU_GE);

  printf("* %d %d\n", m, n);

  // RHS
  int ldb = m_loc;
  int nrhs = 1;
  double* berr;

  if (!(berr = doubleMalloc_dist(nrhs)))
    ABORT("Malloc fails for berr[].");

  superlu_dist_options_t options;
  set_default_options_dist(&options);
  options.Algo3d = YES;
  options.DiagInv = YES;
  options.ReplaceTinyPivot = YES;

  printf("%d %d\n", m, n);
  fflush(stdout);

  dScalePermstruct_t ScalePermstruct;
  dLUstruct_t LUstruct;
  dScalePermstructInit(m, n, &ScalePermstruct);
  dLUstructInit(n, &LUstruct);

  dSOLVEstruct_t SOLVEstruct;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  std::copy(bvec.array().begin(), std::next(bvec.array().begin(), m_loc),
            uvec.mutable_array().begin());

  pdgssvx3d(&options, &A, &ScalePermstruct, uvec.mutable_array().data(), ldb,
            nrhs, &grid, &LUstruct, &SOLVEstruct, berr, &stat, &info);

  if (info)
  { /* Something is wrong */
    printf("ERROR: INFO = %d returned from pdgssvx3d()\n", info);
    fflush(stdout);
  }

  SUPERLU_FREE(berr);
  PStatFree(&stat);
  superlu_gridexit3d(&grid);

  uvec.scatter_fwd();
  return 0;
}
