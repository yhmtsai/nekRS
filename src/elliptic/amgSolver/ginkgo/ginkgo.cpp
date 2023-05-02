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
static int64_t num_local_rows;
static int64_t num_global_rows;
static MPI_Comm dup_comm;
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
  using IndexType = int64_t;
  using ValueType = double;
  auto exec = gko::CudaExecutor::create(deviceID,
                                        gko::ReferenceExecutor::create(),
                                        false,
                                        gko::allocation_mode::device);
  MPI_Comm_dup(comm, &dup_comm);
  int mpi_rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(dup_comm, &mpi_rank);
  MPI_Comm_size(dup_comm, &mpi_size);
  auto matrix_host =
      gko::share(gko::experimental::distributed::Matrix<ValueType, int, IndexType>::create(exec->get_master(),
                                                                                           dup_comm));
  auto rows_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(rows));
  auto cols_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(cols));
  auto vals_view = gko::array<ValueType>::const_view(exec->get_master(), nnz, values);
  gko::array<IndexType> all_num_rows(exec->get_master(), mpi_size + 1);
  num_local_rows = nLocalRows;
  MPI_Allgather(&num_local_rows, 1, MPI_LONG_LONG, all_num_rows.get_data() + 1, 1, MPI_LONG_LONG, dup_comm);
  all_num_rows.get_data()[0] = 0;
  for (int r = 2; r <= mpi_size; r++) {
    all_num_rows.get_data()[r] += all_num_rows.get_data()[r - 1];
  }
  num_global_rows = all_num_rows.get_const_data()[mpi_size];
  auto data =
      gko::device_matrix_data<ValueType, IndexType>(exec->get_master(),
                                                    gko::dim<2>{num_global_rows, num_global_rows},
                                                    gko::detail::array_const_cast(std::move(rows_view)),
                                                    gko::detail::array_const_cast(std::move(cols_view)),
                                                    gko::detail::array_const_cast(std::move(vals_view)));
  auto partition =
      gko::experimental::distributed::Partition<int, IndexType>::build_from_contiguous(exec->get_master(),
                                                                                       all_num_rows);
  matrix_host->read_distributed(data, partition.get());

  // auto coo = gko::matrix::Coo<ValueType, IndexType>::create_const(exec->get_master(),
  //                                                                 gko::dim<2>{nLocalRows, nLocalRows},
  //                                                                 std::move(vals_view),
  //                                                                 std::move(cols_view),
  //                                                                 std::move(rows_view));
  // auto matrix = gko::share(gko::matrix::Csr<ValueType, IndexType>::create(exec));
  auto matrix =
      gko::share(gko::experimental::distributed::Matrix<ValueType, int, IndexType>::create(exec, comm));
  matrix->copy_from(matrix_host.get());
  solver = gko::share(
      gko::solver::Cg<ValueType>::build()
          .with_criteria(gko::stop::Iteration::build().with_max_iters(100u).on(exec))
          .on(exec)
          ->generate(matrix));
}

template <typename ValueType> void GinkgoSolver_solve(ValueType *x, ValueType *rhs)
{
  printf("Ginkgo: solve global %ld local %ld \n", num_global_rows, num_local_rows);
  fflush(stdout);
  // on gpu
  auto exec = solver->get_executor();
  auto x_view = gko::array<ValueType>::view(exec, num_local_rows, x);
  auto rhs_view = gko::array<ValueType>::view(exec, num_local_rows, rhs);
  auto dense_x =
      gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{num_local_rows, 1}, std::move(x_view), 1);
  auto dense_rhs =
      gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{num_local_rows, 1}, std::move(rhs_view), 1);
  auto par_x = gko::experimental::distributed::Vector<ValueType>::create(exec,
                                                                         dup_comm,
                                                                         gko::dim<2>{num_global_rows, 1},
                                                                         gko::give(dense_x).get());
  auto par_rhs = gko::experimental::distributed::Vector<ValueType>::create(exec,
                                                                           dup_comm,
                                                                           gko::dim<2>{num_global_rows, 1},
                                                                           gko::give(dense_rhs).get());
  solver->apply(par_rhs.get(), par_x.get());
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
