// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_matmul_to_fc.hpp"
#include "op/fully_connected.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

ov::intel_cpu::ConvertMatMulToFC::ConvertMatMulToFC() {
    auto activations_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto weights_m = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto matmul_m = ngraph::pattern::wrap_type<ngraph::opset1::MatMul>({ activations_m, weights_m }, ngraph::pattern::has_static_rank());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        // fc_input_a and fc_input_b - are the final inputs that will be set to FullyConnected of GemmIE operations.
        // So in case of adding new operations that takes matmul inputs we need keep update fc_input_a and fc_input_b.
        auto fc_input_a = pattern_map.at(activations_m);
        auto fc_input_b = pattern_map.at(weights_m);

        auto shape_a = fc_input_a.get_partial_shape();
        auto shape_b = fc_input_b.get_partial_shape();
        NGRAPH_CHECK(shape_b.is_static());

        auto rank_a = shape_a.rank().get_length();
        auto rank_b = shape_b.rank().get_length();

        // Transformation to FC is not supported for 1D inputs
        if (rank_a == 1 || rank_b == 1 ||
            rank_a > 3 || rank_b > 3) {
            return false;
        }

        // Check that if second inputs is Constant path and it's shape without ones dimensions has length <= 2
        // we replace MatMul with FullyConnected operation.
        if (!std::dynamic_pointer_cast<ngraph::opset1::Constant>(fc_input_b.get_node_shared_ptr()) ||
            std::count_if(shape_b.begin(), shape_b.end(), [](ngraph::Dimension x) { return x != 1; }) > 2) {
            return false;
        }
        /*
         *  get_aligned_shapes function align two input shapes to have the same size and
         *  the same batch dimensions (last two dimensions are not comparable).
         *  It also checks that dimensions are compatible so in case with two shapes
         *  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
         */

        auto get_aligned_shapes = [shape_a, shape_b, rank_a, rank_b, &matmul]() -> std::tuple<bool, ngraph::PartialShape, ngraph::PartialShape> {
            ngraph::PartialShape shape_a_aligned(shape_a), shape_b_aligned(shape_b);
            size_t max_size = std::max(rank_a, rank_b);
            for (size_t i = 0, cnt = max_size - rank_a; i < cnt; ++i) {
                shape_a_aligned.insert(shape_a_aligned.begin(), 1);
            }
            for (size_t i = 0, cnt = max_size - rank_b; i < cnt; ++i) {
                shape_b_aligned.insert(shape_b_aligned.begin(), 1);
            }

            if (matmul->get_transpose_a() && rank_a != 1) {
                std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
            }
            if (matmul->get_transpose_b()) {
                std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
            }

            for (size_t i = 0; i < max_size - 2; ++i) {
                auto a_dim = shape_a_aligned[i], b_dim = shape_b_aligned[i];
                if (a_dim.is_dynamic()) {
                    if (b_dim == 1) {
                        shape_a_aligned[i] = shape_b_aligned[i] = a_dim;
                    } else {
                        return std::make_tuple(false, ngraph::PartialShape{shape_a_aligned}, ngraph::PartialShape{shape_b_aligned});
                    }
                    continue;
                }
                // both dimensions are static
                if (a_dim != b_dim && a_dim.get_length() > 1 && b_dim.get_length() > 1) {
                    std::ostringstream stream;
                    stream << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
                    throw ngraph::ngraph_error(stream.str());
                }
                size_t max_value = std::max(a_dim.get_length(), b_dim.get_length());
                shape_a_aligned[i] = shape_b_aligned[i] = max_value;
            }
            return std::make_tuple(true, shape_a_aligned, shape_b_aligned);
        };

        /*
         *  create_transpose function return Transpose operation to replace transpose_a or transpose_b
         *  arguments with an operation. In other words in this function we create Transpose operation
         *  with order length equal to output_shape length of given node and fill order with increasing
         *  sequence starting from 0 and replace last two dimension. For example for length = 4  the
         *  order will be [0, 1, 3, 2] that emulates transpose_a or transpose_b attribute.
         */

        auto create_transpose = [this](const ngraph::Output<ngraph::Node>& node, const std::string& transpose_name) {
            auto rank = node.get_partial_shape().rank();
            std::vector<size_t> transpose_order(rank.get_length());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ transpose_order.size() }, transpose_order);
            auto transpose = ngraph::op::util::make_try_fold<ngraph::opset1::Transpose>(node, transpose_const);
            if (!ngraph::is_type<ngraph::opset1::Constant>(transpose)) {
                MatcherPass::register_new_node(transpose);
            }
            transpose->set_friendly_name(transpose_name);
            return transpose;
        };

        ngraph::NodeVector new_ops;
        bool success = true;
        ngraph::PartialShape shape_a_aligned, shape_b_aligned;
        std::tie(success, shape_a_aligned, shape_b_aligned) = get_aligned_shapes();
        if (!success) {
            return false;
        }

        auto aligned_a_rank = shape_a_aligned.rank(), aligned_b_rank = shape_b_aligned.rank();
        if (aligned_a_rank.is_dynamic() || aligned_b_rank.is_dynamic() || aligned_a_rank.get_length() < 2 || aligned_b_rank.get_length() < 2) {
            throw ngraph::ngraph_error("MatMul " + matmul->get_friendly_name() + " shapes are inconsistent.");
        }

        // Transferring from MatMul representation: [B, I, K] * [B, K, O] = [B, I, O]
        // to FullyConnected representation: [I, K] * [K, O] = [I, O]

        // Weights normalization
        if (!matmul->get_transpose_b()) {
            fc_input_b = create_transpose(fc_input_b, matmul->get_friendly_name() + "/transpose_b");
            new_ops.push_back(fc_input_b.get_node_shared_ptr());
        }

        if (rank_b != 2) {
            ngraph::Dimension K = *(shape_b_aligned.rbegin() + 1);
            NGRAPH_CHECK(K.is_static());
            std::vector<int64_t> reshape_shape_values = { -1ll, static_cast<int64_t>(K.get_length()) };
            auto reshape_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, reshape_shape_values);
            fc_input_b = ngraph::op::util::make_try_fold<ngraph::opset1::Reshape>(fc_input_b, reshape_shape, false);
            new_ops.push_back(fc_input_b.get_node_shared_ptr());
        }

        // Input normalization
        if (matmul->get_transpose_a() && rank_a != 1) {
            fc_input_a = create_transpose(fc_input_a, matmul->get_friendly_name() + "/transpose_a");
            new_ops.push_back(fc_input_a.get_node_shared_ptr());
        }

        auto output_rank = matmul->get_output_partial_shape(0).rank();
        // Create FullyConnected
        auto fc = std::make_shared<ov::intel_cpu::FullyConnectedNode>(fc_input_a, fc_input_b, output_rank, matmul->get_output_element_type(0));
        fc->set_friendly_name(matmul->get_friendly_name());
        new_ops.push_back(fc);
        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, fc);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_m, "ConvertMatMulToFC");
    this->register_matcher(m, callback);
}
