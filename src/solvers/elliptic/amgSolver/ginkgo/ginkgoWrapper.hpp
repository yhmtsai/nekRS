#ifndef GINKGO_WRAPPER_H
#define GINKGO_WRAPPER_H

#include <mpi.h>
#include <string>

#ifdef ENABLE_GINKGO
#include <ginkgo/ginkgo.hpp>
#endif

int ginkgoWrapperenabled();

class ginkgoWrapper
{
public:
  ~ginkgoWrapper();

  ginkgoWrapper(const int nLocalRows,
                const int nnz,
                const long long *rows,
                const long long *cols,
                const double *values, /* COO */
                const int null_space,
                const MPI_Comm comm,
                const std::string &backend,
                int deviceID,
                int useFP32,
                bool localOnly,
                bool profiling,
                const std::string &cfg);

  template <typename ValueType> int solve(void *rhs, void *x);

  int solve(void *rhs, void *x);

private:
  MPI_Comm comm_;
  int64_t num_local_rows_;
  int64_t num_global_rows_;
  int use_fp32_;
  bool local_only_;
  bool profiling_;
#ifdef ENABLE_GINKGO
  std::shared_ptr<gko::Executor> exec_;
  std::shared_ptr<gko::LinOp> solver_;
#endif
};

#endif
