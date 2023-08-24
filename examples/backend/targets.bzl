load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # executor runner for XNNPACK Backend and portable kernels.
    runtime.cxx_binary(
        name = "xnn_executor_runner",
        srcs = [],
        deps = [
            "//executorch/examples/executor_runner:executor_runner_lib",
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/kernels/portable:generated_lib_all_ops",
        ],
        define_static_target = True,
        **get_oss_build_kwargs()
    )