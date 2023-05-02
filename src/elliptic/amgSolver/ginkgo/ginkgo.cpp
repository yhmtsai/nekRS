#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "omp.h"

#include <ginkgo/ginkgo.hpp>
#include "ginkgo_elliptic.hpp"

#include <memory>

static std::shared_ptr<gko::LinOp> solver;

void GinkgoSolver_setup(const int nLocalRows,
                        const int nnz,
                        const long long *rows,
                        const long long *cols,
                        const double *values, /* COO */
                        const int null_space,
                        const MPI_Comm comm,
                        int deviceID)
{
  printf("Ginkgo: build solver\n");
  fflush(stdout);
  static_assert(sizeof(int64_t) == 8, "int64_t");
  static_assert(sizeof(long long) == 8, "long long");
  static_assert(sizeof(long) == 8, "long");
  using IndexType = int64_t;
  using ValueType = double;
  auto exec =
      gko::CudaExecutor::create(0, gko::ReferenceExecutor::create(), false, gko::allocation_mode::device);
  auto rows_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(rows));
  auto cols_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(cols));
  auto vals_view = gko::array<ValueType>::const_view(exec->get_master(), nnz, values);
  //   one node nLocalCols = nLocalRows
  auto coo = gko::matrix::Coo<ValueType, IndexType>::create_const(exec->get_master(),
                                                                  gko::dim<2>{nLocalRows, nLocalRows},
                                                                  std::move(vals_view),
                                                                  std::move(cols_view),
                                                                  std::move(rows_view));
  auto matrix = gko::share(gko::matrix::Csr<ValueType, IndexType>::create(exec));
  matrix->copy_from(coo.get());
  solver = gko::share(
      gko::solver::Cg<ValueType>::build()
          .with_criteria(gko::stop::Iteration::build().with_max_iters(100u).on(exec))
          .on(exec)
          ->generate(matrix));
}

template <typename ValueType> void GinkgoSolver_solve(ValueType *x, ValueType *rhs)
{
  printf("Ginkgo: solve\n");
  fflush(stdout);
  // on gpu
  auto exec = solver->get_executor();
  auto n = solver->get_size()[0];
  auto x_view = gko::array<ValueType>::view(exec, n, x);
  auto rhs_view = gko::array<ValueType>::const_view(exec, n, rhs);
  auto dense_x = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{n, 1}, std::move(x_view), 1);
  auto dense_rhs =
      gko::matrix::Dense<ValueType>::create_const(exec, gko::dim<2>{n, 1}, std::move(rhs_view), 1);
  solver->apply(dense_rhs.get(), dense_x.get());
}
void GinkgoSolver_solve(void *x, void *rhs)
{
  // useFP32 -> x, rhs is float
  // !useFP32 -> x, rhs is double
  // it's handled by outsite, we ignore the part -> always double in our side?
  if (true) {
    GinkgoSolver_solve<double>(static_cast<double *>(x), static_cast<double *>(rhs));
  }
}
void GinkgoSolver_free() { solver = nullptr; }
