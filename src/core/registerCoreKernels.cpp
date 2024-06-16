#include <compileKernels.hpp>
#include "LVector.hpp"

void registerCoreKernels()
{
  oogs::registerKernels();

  if (!platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->linAlg = linAlg_t::getInstance();

    std::string kernelName;
    kernelName = "core-copyDfloatToPfloat";
    platform->copyDfloatToPfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyPfloatToDfloat";
    platform->copyPfloatToDfloatKernel = platform->kernelRequests.load(kernelName);

    kernelName = "core-copyDfloatToDouble";
    platform->copyDfloatToDoubleKernel = platform->kernelRequests.load(kernelName);

    return;
  }

  std::string kernelName;
  std::string fileName;

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");
  const std::string section = "core-";
  const std::string extension = platform->serial ? ".c" : ".okl";

  kernelName = "nStagesSum3";
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, platform->kernelInfo);

  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  auto meshProps = platform->kernelInfo;
  meshProps += meshKernelProperties(N);
  const std::string suffix = "Hex3D";

  kernelName = "gradientVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "wGradientVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "wDivergenceVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "divergenceVolume" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "curl" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "filterRT" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "vectorFilterRT" + suffix;
  fileName = oklpath + "/core/" + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  {
    auto prop = meshProps;
    prop["includes"].asArray();
    std::string derivDataFile = oklpath + "/mesh/constantGLLDifferentiationMatrices.h";
    prop["includes"] += derivDataFile.c_str();
    prop["defines/p_inputAdd"] = 0;
    kernelName = "weakLaplacian" + suffix;
    fileName = oklpath + "/core/" + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, prop);
  }

  // register platform kernels
  {
    auto prop = platform->kernelInfo;
    prop["defines/pfloat"] = "double";
    kernelName = "copyDfloatToPfloat";
    fileName = oklpath + "/core/" + kernelName + extension;
    platform->kernelRequests.add(section + "copyDfloatToDouble", fileName, prop);

    prop = platform->kernelInfo;
    kernelName = "copyDfloatToPfloat";
    fileName = oklpath + "/core/" + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);

    kernelName = "copyPfloatToDfloat";
    fileName = oklpath + "/core/" + kernelName + extension;
    platform->kernelRequests.add(section + kernelName, fileName, prop);
  }

  registerLinAlgKernels();
  LVector_t<dfloat>::registerKernels();
  LVector_t<pfloat>::registerKernels();
}
