#ifndef GINKGO_ELLIPTIC_HPP
#define GINKGO_ELLIPTIC_HPP

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

void GinkgoSolver_setup(const int nLocalRows, const int nnz,
              const long long *rows, const long long *cols, const double *values, /* COO */ 
              const int null_space, const MPI_Comm comm, int deviceID);
void GinkgoSolver_solve(void *x, void *rhs);
void GinkgoSolver_free();

#ifdef __cplusplus
}
#endif

#endif
