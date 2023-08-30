#include <stdio.h>
#include <string>
#include <mpi.h>
#include <fstream>

#include "ginkgoWrapper.hpp"

#ifdef ENABLE_GINKGO

#include <nlohmann/json.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/config.hpp>

void json_parser(gko::config::pnode &ptree, const nlohmann::json &dom)
{
  if (dom.is_array()) {
    int num = dom.size();
    ptree.get_array().resize(num);
    for (int i = 0; i < num; i++) {
      json_parser(ptree.at(i), dom[i]);
    }
  } else if (dom.is_object()) {
    auto &list = ptree.get_list();
    for (auto &m : dom.items()) {
      json_parser(list[m.key()], m.value());
    }
  } else {
    if (dom.is_number_integer()) {
      ptree = gko::config::pnode{dom.template get<long long int>()};
    } else if (dom.is_boolean()) {
      ptree = gko::config::pnode{dom.template get<bool>()};
    } else if (dom.is_number_float()) {
      ptree = gko::config::pnode{dom.template get<double>()};
    } else if (dom.is_string()) {
      ptree = gko::config::pnode{std::string(dom.template get<std::string>())};
    } else {
      ptree = gko::config::pnode{};
    }
  }
}

ginkgoWrapper::ginkgoWrapper(const int nLocalRows,
                             const int nnz,
                             const long long *rows,
                             const long long *cols,
                             const double *values, /* COO */
                             const int nullspace,
                             const MPI_Comm comm,
                             int deviceID,
                             int useFP32,
                             int MPI_DIRECT,
                             const std::string &cfg)
{
  printf("Ginkgo: build solver\n");
  fflush(stdout);
  using IndexType = int64_t;
  using ValueType = double;
  std::shared_ptr<const gko::Executor> exec =
      gko::CudaExecutor::create(deviceID, gko::ReferenceExecutor::create());

  MPI_Comm_dup(comm, &comm_);

  int mpi_rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(comm_, &mpi_rank);
  MPI_Comm_size(comm_, &mpi_size);

  gko::array<IndexType> all_num_rows(exec->get_master(), mpi_size + 1);
  num_local_rows_ = nLocalRows;
  MPI_Allgather(&num_local_rows_, 1, MPI_LONG_LONG, all_num_rows.get_data() + 1, 1, MPI_LONG_LONG, comm_);
  all_num_rows.get_data()[0] = 0;
  for (int r = 2; r <= mpi_size; r++) {
    all_num_rows.get_data()[r] += all_num_rows.get_data()[r - 1];
  }
  num_global_rows_ = all_num_rows.get_const_data()[mpi_size];

  // use device_matrix_data to handle coo data
  auto rows_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(rows));
  auto cols_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(cols));
  auto vals_view = gko::array<ValueType>::const_view(exec->get_master(), nnz, values);
  auto data =
      gko::device_matrix_data<ValueType, IndexType>(exec->get_master(),
                                                    gko::dim<2>{num_global_rows_, num_global_rows_},
                                                    gko::detail::array_const_cast(std::move(rows_view)),
                                                    gko::detail::array_const_cast(std::move(cols_view)),
                                                    gko::detail::array_const_cast(std::move(vals_view)));
  // create the partition
  auto partition =
      gko::experimental::distributed::Partition<int, IndexType>::build_from_contiguous(exec->get_master(),
                                                                                       all_num_rows);

  auto matrix_host = gko::share(
      gko::experimental::distributed::Matrix<ValueType, int, IndexType>::create(exec->get_master(), comm_));
  // create the matrix
  matrix_host->read_distributed(data, partition.get());
  // copy to device
  auto matrix =
      gko::share(gko::experimental::distributed::Matrix<ValueType, int, IndexType>::create(exec, comm));
  matrix->copy_from(matrix_host.get());

  if (cfg.size()) {
    gko::config::pnode config;
    std::ifstream f(cfg);
    json_parser(config, cfg);

    gko::config::registry reg(gko::config::generate_config_map());
    solver_ = gko::share(gko::config::build_from_config(config, reg, exec, {"double", "int"})->generate(matrix));
  } else {
    solver_ = gko::share(gko::solver::Cg<ValueType>::build()
                             .with_criteria(gko::stop::Iteration::build().with_max_iters(100u).on(exec))
                             .on(exec)
                             ->generate(matrix));
  }
}

int ginkgoWrapper::solve(void *rhs, void *x)
{
  // vector always use float
  using ValueType = float;
  printf("Ginkgo: solve global %ld local %ld \n", num_global_rows_, num_local_rows_);
  fflush(stdout);
  // on gpu
  auto exec = solver_->get_executor();
  auto x_view = gko::array<ValueType>::view(exec, num_local_rows_, static_cast<ValueType *>(x));
  auto rhs_view = gko::array<ValueType>::view(exec, num_local_rows_, static_cast<ValueType *>(rhs));
  auto dense_x =
      gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{num_local_rows_, 1}, std::move(x_view), 1);
  auto dense_rhs =
      gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{num_local_rows_, 1}, std::move(rhs_view), 1);
  auto par_x = gko::experimental::distributed::Vector<ValueType>::create(exec,
                                                                         comm_,
                                                                         gko::dim<2>{num_global_rows_, 1},
                                                                         gko::give(dense_x));
  auto par_rhs = gko::experimental::distributed::Vector<ValueType>::create(exec,
                                                                           comm_,
                                                                           gko::dim<2>{num_global_rows_, 1},
                                                                           gko::give(dense_rhs));
  solver_->apply(par_rhs.get(), par_x.get());

  return 0;
}

ginkgoWrapper::~ginkgoWrapper()
{
  MPI_Comm_free(&comm_);
}

int ginkgoWrapperenabled()
{
  return 1;
}

#else
ginkgoWrapper::ginkgoWrapper(const int nLocalRows,
                             const int nnz,
                             const long long *rows,
                             const long long *cols,
                             const double *values, /* COO */
                             const int nullspace,
                             const MPI_Comm comm,
                             int deviceID,
                             int useFP32,
                             int MPI_DIRECT,
                             const std::string &cfg)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    printf("ERROR: Recompile with Ginkgo support!\n");
}

int ginkgoWrapper::solve(void *x, void *rhs)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    printf("ERROR: Recompile with Ginkgo support!\n");
  return 1;
}

ginkgoWrapper::~ginkgoWrapper() {}

int ginkgoWrapperenabled()
{
  return 0;
}
#endif
