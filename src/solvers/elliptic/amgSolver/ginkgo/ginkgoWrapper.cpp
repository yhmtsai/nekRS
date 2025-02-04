#include <stdio.h>
#include <string>
#include <mpi.h>
#include <fstream>
#include <iostream>

#include "ginkgoWrapper.hpp"

#ifdef ENABLE_GINKGO

#include <ginkgo/extensions/config/json_config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/config.hpp>

ginkgoWrapper::ginkgoWrapper(const int nLocalRows,
                             const int nnz,
                             const long long *rows,
                             const long long *cols,
                             const double *values, /* COO */
                             const int nullspace,
                             const MPI_Comm comm,
                             const std::string &backend,
                             int deviceID,
                             int useFP32,
                             bool localOnly,
                             bool profiling,
                             bool useHalf,
                             const std::string &cfg)
{
  using IndexType = int64_t;
  using ValueType = double;
  static_assert(sizeof(IndexType) == sizeof(long long));
  // CPU|CUDA|HIP|DPCPP
  // CPU: Serial/OpenMP?
  std::map<std::string, std::function<std::shared_ptr<gko::Executor>(int)>> executor_factory{
      {"CPU",
       [](int device_id) -> std::shared_ptr<gko::Executor> {
         static const std::string not_compiled_tag = "not compiled";
         auto version = gko::version_info::get();
         if (version.omp_version.tag != not_compiled_tag) {
           return gko::OmpExecutor::create();
         } else {
           return gko::ReferenceExecutor::create();
         }
       }},
      {"CUDA",
       [](int device_id) { return gko::CudaExecutor::create(device_id, gko::ReferenceExecutor::create()); }},
      {"HIP",
       [](int device_id) { return gko::HipExecutor::create(device_id, gko::ReferenceExecutor::create()); }},
      {"DPCPP", [](int device_id) {
         return gko::DpcppExecutor::create(device_id, gko::ReferenceExecutor::create());
       }}};
  exec_ = executor_factory.at(backend)(deviceID);
  auto exec = exec_;
  use_fp32_ = useFP32;
  local_only_ = localOnly;
  profiling_ = profiling;
  use_half_ = useHalf;
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
  if (mpi_rank == 0) {
    printf("Ginkgo: build solver %s - %s - localOnly %d - profiling %d - useFP32 %d - useHalf %d\n",
           backend.c_str(),
           cfg.c_str(),
           local_only_,
           profiling_,
           use_fp32_,
           use_half_);
    std::ifstream();
    std::cout << "===== config =====" << std::endl;
    std::ifstream f(cfg);
    std::cout << f.rdbuf() << std::endl;
    std::cout << "===== config =====" << std::endl;
  }
  printf("Ginkgo: solve global %ld local %ld (rank id %d) nnz %d\n",
         num_global_rows_,
         num_local_rows_,
         mpi_rank,
         nnz);
  fflush(stdout);

  // use device_matrix_data to handle coo data
  auto rows_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(rows));
  auto cols_view =
      gko::array<IndexType>::const_view(exec->get_master(), nnz, reinterpret_cast<const IndexType *>(cols));
  auto vals_view = gko::array<ValueType>::const_view(exec->get_master(), nnz, values);
  std::cout << "first elem of rank " << mpi_rank << " " << rows[0] << ", " << cols[0] << ", " << values[0]
            << std::endl;
  auto data =
      gko::device_matrix_data<ValueType, IndexType>(exec->get_master(),
                                                    gko::dim<2>{num_global_rows_, num_global_rows_},
                                                    gko::detail::array_const_cast(std::move(rows_view)),
                                                    gko::detail::array_const_cast(std::move(cols_view)),
                                                    gko::detail::array_const_cast(std::move(vals_view)));
  // create the partition
  auto partition = gko::share(
      gko::experimental::distributed::Partition<int, IndexType>::build_from_contiguous(exec->get_master(),
                                                                                       all_num_rows));
  if (mpi_rank == 0) {
    auto num_parts = partition->get_num_parts();
    std::cout << "num_parts: " << num_parts << " should be the same as " << mpi_size << std::endl;
    for (int i = 0; i < num_parts; i++) {
      std::cout << "rank:" << i << " " << partition->get_part_size(i) << std::endl;
    }
  }
  MPI_Barrier(comm_);

  auto matrix_host = gko::share(
      gko::experimental::distributed::Matrix<ValueType, int, IndexType>::create(exec->get_master(), comm_));
  // create the matrix
  matrix_host->read_distributed(data, partition);
  // copy to device
  auto matrix =
      gko::share(gko::experimental::distributed::Matrix<ValueType, int, IndexType>::create(exec, comm_));
  matrix->copy_from(matrix_host.get());
  std::cout << "rank:" << mpi_rank << " local size " << matrix->get_local_matrix()->get_size()
            << " non_local size " << matrix->get_non_local_matrix()->get_size() << std::endl;

  std::shared_ptr<gko::LinOp> linop;
  if (local_only_) {
    auto local_matrix = matrix->get_local_matrix();
    half_rhs_ = gko::matrix::Dense<gko::half>::create(exec);
    half_x_ = gko::matrix::Dense<gko::half>::create(exec);
    if (use_half_) {
      auto matrix_half = gko::share(gko::matrix::Csr<gko::half, int>::create(exec));
      matrix_half->copy_from(local_matrix.get());
      linop = matrix_half;
    } else if (use_fp32_) {
      auto matrix_float = gko::share(gko::matrix::Csr<float, int>::create(exec));
      matrix_float->copy_from(local_matrix.get());
      linop = matrix_float;
    } else {
      auto matrix_double = gko::share(gko::matrix::Csr<double, int>::create(exec));
      matrix_double->copy_from(local_matrix.get());
      linop = matrix_double;
    }
  } else {
    half_rhs_ = gko::experimental::distributed::Vector<gko::half>::create(exec, comm_);
    half_x_ = gko::experimental::distributed::Vector<gko::half>::create(exec, comm_);
    if (use_half_) {
      auto matrix_half =
          gko::share(gko::experimental::distributed::Matrix<gko::half, int, IndexType>::create(exec, comm_));
      matrix_half->copy_from(matrix.get());
      linop = matrix_half;
    } else if (use_fp32_) {
      auto matrix_float =
          gko::share(gko::experimental::distributed::Matrix<float, int, IndexType>::create(exec, comm_));
      matrix_float->copy_from(matrix.get());
      linop = matrix_float;
    } else {
      linop = matrix;
    }
  }
  if (mpi_rank == 0) {
    std::cout << "Rank 0 system_matrix size: " << linop->get_size() << std::endl;
  }
  if (cfg.size()) {
    auto config = gko::ext::config::parse_json_file(cfg);

    gko::config::registry reg;
    auto td = use_half_ ? gko::config::make_type_descriptor<gko::half, int>()
                        : (use_fp32_ ? gko::config::make_type_descriptor<float, int>()
                                     : gko::config::make_type_descriptor<double, int>());
    solver_ = gko::share(gko::config::parse(config, reg, td).on(exec)->generate(linop));
  } else {
    if (use_half_) {
      solver_ = gko::share(gko::solver::Cg<gko::half>::build()
                               .with_criteria(gko::stop::Iteration::build().with_max_iters(100u).on(exec))
                               .on(exec)
                               ->generate(linop));
    } else if (use_fp32_) {
      solver_ = gko::share(gko::solver::Cg<float>::build()
                               .with_criteria(gko::stop::Iteration::build().with_max_iters(100u).on(exec))
                               .on(exec)
                               ->generate(linop));
    } else {
      solver_ = gko::share(gko::solver::Cg<ValueType>::build()
                               .with_criteria(gko::stop::Iteration::build().with_max_iters(100u).on(exec))
                               .on(exec)
                               ->generate(linop));
    }
  }
}

template <typename ValueType> int ginkgoWrapper::solve(void *rhs, void *x)
{
  auto exec = exec_;
  auto x_view = gko::array<ValueType>::view(exec, num_local_rows_, static_cast<ValueType *>(x));
  auto rhs_view = gko::array<ValueType>::view(exec, num_local_rows_, static_cast<ValueType *>(rhs));
  auto dense_x =
      gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{num_local_rows_, 1}, std::move(x_view), 1);
  auto dense_rhs =
      gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{num_local_rows_, 1}, std::move(rhs_view), 1);

  auto run = [&]() {
    if (local_only_) {
      if (use_half_) {
        half_x_->copy_from(dense_x.get());
        half_rhs_->copy_from(dense_rhs.get());
        solver_->apply(half_rhs_.get(), half_x_.get());
        dense_x->copy_from(half_x_.get());
      } else {
        solver_->apply(dense_rhs.get(), dense_x.get());
      }
    } else {
      auto par_x = gko::experimental::distributed::Vector<ValueType>::create(exec,
                                                                             comm_,
                                                                             gko::dim<2>{num_global_rows_, 1},
                                                                             gko::give(dense_x));
      auto par_rhs =
          gko::experimental::distributed::Vector<ValueType>::create(exec,
                                                                    comm_,
                                                                    gko::dim<2>{num_global_rows_, 1},
                                                                    gko::give(dense_rhs));
      if (use_half_) {
        half_x_->copy_from(par_x.get());
        half_rhs_->copy_from(par_rhs.get());
        solver_->apply(half_rhs_.get(), half_x_.get());
        par_x->copy_from(half_x_.get());
      } else {
        solver_->apply(par_rhs.get(), par_x.get());
      }
    }
  };

  static int step = 0;
  if (profiling_) {
    auto profiler = gko::log::ProfilerHook::create_for_executor(exec);
    exec->add_logger(profiler);
    std::string name = "Ginkgo_";
    name += std::to_string(step);
    auto range = profiler->user_range(name.c_str());
    run();
    exec->remove_logger(profiler);
  } else {
    run();
  }
  step++;
  return 0;
}

int ginkgoWrapper::solve(void *rhs, void *x)
{
  if (use_fp32_) {
    return this->template solve<float>(rhs, x);
  } else {
    return this->template solve<double>(rhs, x);
  }
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
                             const std::string &backend,
                             int deviceID,
                             int useFP32,
                             bool localOnly,
                             bool profiling,
                             bool useHalf,
                             const std::string &cfg)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("ERROR: Recompile with Ginkgo support!\n");
  }
}

template <ValueType> int ginkgoWrapper::solve(void *rhs, void *x)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("ERROR: Recompile with Ginkgo support!\n");
  }
  return 1;
}

int ginkgoWrapper::solve(void *rhs, void *x)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("ERROR: Recompile with Ginkgo support!\n");
  }
  return 1;
}

ginkgoWrapper::~ginkgoWrapper() {}

int ginkgoWrapperenabled()
{
  return 0;
}
#endif
