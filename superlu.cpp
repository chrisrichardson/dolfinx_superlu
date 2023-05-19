
#include "superlu_ddefs.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

using namespace dolfinx;

int superlu_solver(MPI_Comm comm, la::MatrixCSR<double>& Amat,
                   const la::Vector<double>& bvec, la::Vector<double>& uvec)
{
  int size = dolfinx::MPI::size(comm);

  int nprow = size;
  int np = 1;
  while (nprow % 2 == 0)
  {
    np *= 2;
    nprow /= 2;
  }

  gridinfo3d_t grid;
  superlu_gridinit3d(comm, nprow, 1, np, &grid);

  // Global size
  int m = Amat.index_map(0)->size_global();
  int n = Amat.index_map(1)->size_global();
  if (m != n)
    throw std::runtime_error("Can't solve non-square system");

  // Number of local rows
  int m_loc = Amat.num_owned_rows();

  // First row
  int first_row = Amat.index_map(0)->local_range()[0];

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

  dCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                 Amat.values().data(), cols.data(),
                                 row_ptr.data(), SLU_NR_loc, SLU_D, SLU_GE);

  // RHS
  int ldb = m_loc;
  int nrhs = 1;
  std::vector<double> berr(nrhs);

  superlu_dist_options_t options;
  set_default_options_dist(&options);
  options.Algo3d = YES;
  options.DiagInv = YES;
  options.ReplaceTinyPivot = YES;

  dScalePermstruct_t ScalePermstruct;
  dLUstruct_t LUstruct;
  dScalePermstructInit(m, n, &ScalePermstruct);
  dLUstructInit(n, &LUstruct);

  dSOLVEstruct_t SOLVEstruct;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU replaces RHS with solution)
  std::copy(bvec.array().begin(), std::next(bvec.array().begin(), m_loc),
            uvec.mutable_array().begin());

  pdgssvx3d(&options, &A, &ScalePermstruct, uvec.mutable_array().data(), ldb,
            nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

  if (info)
  { /* Something is wrong */
    printf("ERROR: INFO = %d returned from pdgssvx3d()\n", info);
    fflush(stdout);
  }

  PStatFree(&stat);
  superlu_gridexit3d(&grid);

  // Update ghosts in u
  uvec.scatter_fwd();
  return 0;
}
