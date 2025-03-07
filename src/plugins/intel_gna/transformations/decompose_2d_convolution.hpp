// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ie_precision.hpp>

namespace GNAPluginNS {

/**
 * @brief Decompose a 2D convolution, wrapped with transposes,
 * to a set of valid 1D convolutions with padding added in front of the set:
 *
 *                                                  Padding
 *                                                     |
 *   Transpose (NHWC -> NCHW)               Transpose (NHWC -> NCHW)
 *              |                                      |
 *   Convolution with padding                  Valid convolution
 *              |                                      |
 *   Broadcast Bias (optional)              Broadcast Bias (optional)
 *              |                                      |
 *    Max Pooling (optional)                 Max Pooling (optional)
 *              |                                      |
 * Activation Function (optional)       Activation Function (optional)
 *              |                                      |
 *   Transpose (NCHW -> NHWC)               Transpose (NCHW -> NHWC)
 *
 */
class Decompose2DConv : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Decompose2DConv", "0");
    Decompose2DConv(const std::string& gnaCompileTarget, const InferenceEngine::Precision& gnaPrecision);
};

/**
 * @brief Decomopose a 2D convolution wrapped with transposes, with bias after trailing transpose,
 * to a set of valid 1D convolutions with padding added in front of the set:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding                Valid convolution
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *
 */
class Decompose2DConvTransposedWithBias : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Decompose2DConvTransposedWithBias", "0");
    Decompose2DConvTransposedWithBias(const std::string& gnaCompileTarget, const InferenceEngine::Precision& gnaPrecision);
};

/**
 * @brief Decomopose a 2D convolution wrapped with transposes, with bias
 * to a set of valid 1D convolutions with padding added in front of the set:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding                Valid convolution
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *            |                                    |
 *   Activation Function                  Activation Function
 * 
 */
class Decompose2DConvTransposedWithBiasAF : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Decompose2DConvTransposedWithBiasAF", "0");
    Decompose2DConvTransposedWithBiasAF(const std::string& gnaCompileTarget, const InferenceEngine::Precision& gnaPrecision);
};

} // namespace GNAPluginNS
