
#include "superlu_ddefs.h"
#include "superlu_sdefs.h"
#include "superlu_zdefs.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <iostream>

using namespace dolfinx;

template <typename T>
int superlu_solver(MPI_Comm comm, const la::MatrixCSR<T>& Amat,
                   const la::Vector<T>& bvec, la::Vector<T>& uvec)
{
  int size = dolfinx::MPI::size(comm);

  int nprow = size;
  int npcol = 1;
  int np = 1;
  // while (nprow % 2 == 0)
  // {
  //   np *= 2;
  //   nprow /= 2;
  // }

  gridinfo3d_t grid;
  superlu_gridinit3d(comm, nprow, npcol, np, &grid);

  // Global size
  int m = Amat.index_map(0)->size_global();
  int n = Amat.index_map(1)->size_global();
  if (m != n)
    throw std::runtime_error("Can't solve non-square system");

  // Number of local rows
  int m_loc = Amat.num_owned_rows();

  // First row
  int first_row = Amat.index_map(0)->local_range()[0];

  // Local number of non-zeros
  int nnz_loc = Amat.row_ptr()[m_loc];
  std::vector<int_t> cols(nnz_loc);
  std::vector<int_t> rowptr(m_loc + 1);

  // Copy row_ptr from int64
  std::copy(Amat.row_ptr().begin(),
            std::next(Amat.row_ptr().begin(), m_loc + 1), rowptr.begin());

  // Convert local to global indices (and cast to int_t)
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), std::next(Amat.cols().begin(), nnz_loc),
                 cols.begin(),
                 [&](std::int64_t local_index)
                 { return global_col_indices[local_index]; });

  SuperMatrix A;
  if constexpr (std::is_same_v<T, double>)
  {
    auto Amatdata = const_cast<double*>(Amat.values().data());
    dCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.data(), rowptr.data(),
                                   SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    auto Amatdata = const_cast<float*>(Amat.values().data());
    sCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.data(), rowptr.data(),
                                   SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    auto Amatdata = const_cast<std::complex<double>*>(Amat.values().data());
    zCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   reinterpret_cast<doublecomplex*>(Amatdata),
                                   cols.data(), rowptr.data(), SLU_NR_loc,
                                   SLU_Z, SLU_GE);
  }

  // RHS
  int ldb = m_loc;
  int nrhs = 1;

  superlu_dist_options_t options;
  set_default_options_dist(&options);
  options.Algo3d = YES;
  options.DiagInv = YES;
  options.ReplaceTinyPivot = YES;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU replaces RHS with solution)
  std::copy(bvec.array().begin(), std::next(bvec.array().begin(), m_loc),
            uvec.mutable_array().begin());

  if constexpr (std::is_same_v<T, double>)
  {
    std::vector<T> berr(nrhs);
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dScalePermstructInit(m, n, &ScalePermstruct);
    dLUstructInit(n, &LUstruct);
    dSOLVEstruct_t SOLVEstruct;

    pdgssvx3d(&options, &A, &ScalePermstruct, uvec.mutable_array().data(), ldb,
              nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
    dSolveFinalize(&options, &SOLVEstruct);
    dDestroy_A3d_gathered_on_2d(&SOLVEstruct, &grid);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    std::vector<T> berr(nrhs);
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sScalePermstructInit(m, n, &ScalePermstruct);
    sLUstructInit(n, &LUstruct);
    sSOLVEstruct_t SOLVEstruct;

    psgssvx3d(&options, &A, &ScalePermstruct, uvec.mutable_array().data(), ldb,
              nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

    sSolveFinalize(&options, &SOLVEstruct);
    sLUstructFree(&LUstruct);
    sScalePermstructFree(&ScalePermstruct);
    sDestroy_A3d_gathered_on_2d(&SOLVEstruct, &grid);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    std::vector<double> berr(nrhs);
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zScalePermstructInit(m, n, &ScalePermstruct);
    zLUstructInit(n, &LUstruct);
    zSOLVEstruct_t SOLVEstruct;

    pzgssvx3d(&options, &A, &ScalePermstruct,
              reinterpret_cast<doublecomplex*>(uvec.mutable_array().data()),
              ldb, nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat,
              &info);

    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
    zSolveFinalize(&options, &SOLVEstruct);
    zDestroy_A3d_gathered_on_2d(&SOLVEstruct, &grid);
  }
  Destroy_SuperMatrix_Store_dist(&A);

  if (info)
  {
    std::cout << "ERROR: INFO = " << info << " returned from p*gssvx3d()\n"
              << std::flush;
  }

  PStatPrint(&options, &stat, &grid.grid2d);
  PStatFree(&stat);

  superlu_gridexit3d(&grid);

  // Update ghosts in u
  uvec.scatter_fwd();
  return info;
}

// Explicit instantiation
template int superlu_solver(MPI_Comm comm, const la::MatrixCSR<double>& Amat,
                            const la::Vector<double>& bvec,
                            la::Vector<double>& uvec);

template int superlu_solver(MPI_Comm comm, const la::MatrixCSR<float>& Amat,
                            const la::Vector<float>& bvec,
                            la::Vector<float>& uvec);

template int superlu_solver(MPI_Comm comm,
                            const la::MatrixCSR<std::complex<double>>& Amat,
                            const la::Vector<std::complex<double>>& bvec,
                            la::Vector<std::complex<double>>& uvec);
