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

  template <typename ValueType>
  ginkgoWrapper(const int nLocalRows,
                const int nnz,
                const long long *rows,
                const long long *cols,
                const ValueType *values, /* COO */
                const int null_space,
                const MPI_Comm comm,
                const std::string &backend,
                int deviceID,
                int useFP32,
                bool localOnly,
                const std::string &cfg);

  template <typename ValueType> int solve(void *rhs, void *x);

  int solve(void *rhs, void *x);

private:
  MPI_Comm comm_;
  gko::size_type num_local_rows_;
  gko::size_type num_global_rows_;
  int use_fp32_;
  bool local_only_;
#ifdef ENABLE_GINKGO
  std::shared_ptr<gko::LinOp> solver_;
#endif
};

#endif
