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
                int deviceID,
                int useFP32,
                int MPIDIRECT,
                const std::string &cfg);

  int solve(void *rhs, void *x);

private:
  MPI_Comm comm_;
  int64_t num_local_rows_;
  int64_t num_global_rows_;
#ifdef ENABLE_GINKGO
  std::shared_ptr<gko::LinOp> solver_;
#endif
};

#endif
