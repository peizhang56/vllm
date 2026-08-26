#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <optional>

// Torch 2.10 stable::from_blob has no deleter overload. The Python wrapper
// pins and retains the CPU backing tensor on the returned accelerator view.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = torch::headeronly::Layout::Strided;
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  STD_TORCH_CHECK(torch::stable::detail::to<bool>(is_pinned_stack[0]),
                  "Torch 2.10 UVA compatibility requires pinned CPU memory");

  void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                  cudaGetErrorString(err));

  return torch::stable::from_blob(device_ptr, cpu_tensor.sizes(),
                                  cpu_tensor.strides(), cuda_dev, dtype, 0,
                                  layout);
}
