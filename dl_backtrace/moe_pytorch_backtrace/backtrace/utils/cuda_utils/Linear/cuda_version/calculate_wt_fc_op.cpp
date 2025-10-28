#include <torch/extension.h>
#include "utils.h"
#include <limits>
#include <map>
#include <string>

torch::Tensor calculate_wt_fc_interface(
    const torch::Tensor& row_specific_weights,
    const torch::Tensor& input_activations,
    const torch::Tensor& weights_matrix,
    const torch::Tensor& bias_vector,
    const bool has_lower_bound,
    const c10::optional<float>& lower_threshold,
    const bool has_upper_bound,
    const c10::optional<float>& upper_threshold,
    const bool is_non_mono,
    const int activation_func
) {
    // std::cout << "--- C++ Input Shapes ---" << std::endl;
    // std::cout << "row_specific_weights.shape: " << row_specific_weights.sizes() << "device: " << row_specific_weights.device() << std::endl;
    // std::cout << "input_activations.shape: " << input_activations.sizes() << "device: " << input_activations.device() << std::endl;
    // std::cout << "weights_matrix.shape: " << weights_matrix.sizes() << "device: " << weights_matrix.device() << std::endl;
    // std::cout << "bias_vector.shape: " << bias_vector.sizes() << "device: " << bias_vector.device() << std::endl;

    // --- Tensor Checks (already present and good) ---
    TORCH_CHECK(row_specific_weights.dim() == 1, "row_specific_weights must be 1D");
    TORCH_CHECK(input_activations.dim() == 1, "input_activations must be 1D");
    TORCH_CHECK(weights_matrix.dim() == 2, "weights_matrix must be 2D");
    TORCH_CHECK(bias_vector.dim() == 1, "bias_vector must be 1D");
    
    int D_in = weights_matrix.size(1);
    int D_out = weights_matrix.size(0);

    // TORCH_CHECK(row_specific_weights.sizes()[0] == D_out, "row_specific_weights size mismatch with D_in");
    // TORCH_CHECK(input_activations.sizes()[0] == D_in, "input_activations size mismatch with D_in");
    // TORCH_CHECK(bias_vector.sizes()[0] == D_out, "bias_vector size mismatch with D_in");

    // --- Activation Parameter Checks ---
    if (has_lower_bound && has_upper_bound) {
        TORCH_CHECK(lower_threshold <= upper_threshold,
                    "If both bounds are active, lower_threshold (", lower_threshold,
                    ") must be less than or equal to upper_threshold (", upper_threshold, ")");
    }

    // Check for valid activation_func_enum values (adjust based on your kernel's switch statement)
    // Example: 0 for identity, 1 for ReLU, 2 for Sigmoid
    TORCH_CHECK(activation_func >= 0 && activation_func <= 2, // Assuming 0, 1, 2 are your valid enums
                "Invalid activation_func: ", activation_func, ". Expected 0 (identity), 1 (ReLU), or 2 (Sigmoid).");

    if (is_non_mono) {
        // Non-monotonic logic might only make sense with certain activation functions.
        // For example, if it's only for identity, ReLU, Sigmoid in your kernel:
        TORCH_CHECK(activation_func >= 0 && activation_func <= 2,
                    "is_non_mono is true, but activation_func (", activation_func,
                    ") is not one that supports non-monotonic behavior in this kernel (expected 0, 1, or 2).");
        // Or, if non-monotonicity itself implies a specific activation type or is incompatible with others, add checks.
    }

    // Add any other logical checks that make sense for your parameters.
    // For instance, if lower_threshold or upper_threshold should not be NaN or Inf:
    // TORCH_CHECK(std::isfinite(lower_threshold), "lower_threshold must be a finite number.");
    // TORCH_CHECK(std::isfinite(upper_threshold), "upper_threshold must be a finite number.");
    // (Requires #include <cmath> for std::isfinite)


    // Call the CUDA launcher function
    // Ensure its signature in utils.h and .cu definition matches these parameters
    return calculate_wt_fc_cuda(
        row_specific_weights,
        input_activations,
        weights_matrix,
        bias_vector,
        has_lower_bound,
        lower_threshold,
        has_upper_bound,
        upper_threshold,
        is_non_mono,
        activation_func // Pass the validated enum
    );
}

// PYBIND11_MODULE definition remains the same
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_wt_fc_interface", &calculate_wt_fc_interface, "Fused Weighted Fully Connected Layer with activation controls",
        py::arg("row_specific_weights"),
        py::arg("input_activations"),
        py::arg("weights_matrix"),
        py::arg("bias_vector"),
        py::arg("has_lower_bound"),
        py::arg("lower_threshold"),
        py::arg("has_upper_bound"),
        py::arg("upper_threshold"),
        py::arg("is_non_mono"),
        py::arg("activation_func")
    );
}
