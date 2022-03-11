// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ReshapeSequenceFusion, "ReshapeSequenceFusion", 0);

namespace {
bool has_valid_pattern(const ov::Output<ov::Node>& node_out) {
    const auto const_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node_out.get_node_shared_ptr());
    if (!const_node) {
        // Lower bound of the value
        auto lb = ngraph::evaluate_lower_bound(node_out);
        if (!lb)
            return false;
        const auto lb_const_node = std::make_shared<ngraph::opset8::Constant>(lb);
        const auto& lb_values = lb_const_node->cast_vector<int64_t>();

        // The pattern is valid if all lower bound values are higher than zero (not a special number)
        // or if the lower and upper bounds values are a sign of full dynamism
        const bool lb_has_special_val = std::any_of(lb_values.cbegin(), lb_values.cend(), [](int64_t value) {
            return value < 1;
        });
        if (!lb_has_special_val)
            return true;

        // Upper bound of the value
        auto ub = ngraph::evaluate_upper_bound(node_out);
        if (!ub)
            return false;

        const auto ub_const_node = std::make_shared<ngraph::opset8::Constant>(ub);
        const auto& ub_values = ub_const_node->cast_vector<int64_t>();
        if (lb_values.size() != ub_values.size())
            return false;

        // Check if zero values are paired with max value as a sign of full dynamism
        const int64_t ub_max = node_out.get_element_type() == ov::element::i32 ? std::numeric_limits<int32_t>::max()
                                                                               : std::numeric_limits<int64_t>::max();
        const auto mismatch_iters = std::mismatch(lb_values.cbegin(),
                                                  lb_values.cend(),
                                                  ub_values.cbegin(),
                                                  [ub_max](int64_t lb_val, int64_t ub_val) {
                                                      return lb_val > 0 || (lb_val == 0 && ub_val == ub_max);
                                                  });
        return mismatch_iters.first == lb_values.cend();
    }
    const auto& values = const_node->cast_vector<int64_t>();
    // We can not fuse Reshapes if their pattern values have special numbers like -1 and 0
    return std::all_of(values.cbegin(), values.cend(), [](int64_t value) {
        return value > 0;
    });
}
}  // namespace

ngraph::pass::ReshapeSequenceFusion::ReshapeSequenceFusion(bool use_shape_for_elimination) {
    MATCHER_SCOPE(ReshapeSequenceFusion);
    auto reshape_input = pattern::any_input();
    auto reshape_a_pattern = pattern::wrap_type<opset8::Constant>();
    auto reshape_a =
        pattern::wrap_type<opset8::Reshape>({reshape_input, reshape_a_pattern}, pattern::consumers_count(1));
    auto reshape_b_pattern = pattern::any_input();
    auto reshape_b = pattern::wrap_type<opset8::Reshape>({reshape_a, reshape_b_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto input = pattern_map.at(reshape_input);
        auto reshape = m.get_match_root();

        auto pattern_a = pattern_map.at(reshape_a_pattern);
        auto pattern_b = pattern_map.at(reshape_b_pattern);
        // skip reshapes which patterns contain special numbers like -1 or 0
        if (!has_valid_pattern(pattern_a) || !has_valid_pattern(pattern_b)) {
            return false;
        }

        // vector of nodes which runtime info must be copied
        NodeVector nodes{pattern_map.at(reshape_a).get_node_shared_ptr(), reshape};
        while (std::dynamic_pointer_cast<opset8::Reshape>(input.get_node_shared_ptr())) {
            auto node = input.get_node_shared_ptr();
            if (!has_valid_pattern(node->get_input_node_shared_ptr(1)) || input.get_target_inputs().size() != 1) {
                break;
            }
            nodes.push_back(node);
            input = node->input_value(0);
        }

        // remove redundant reshapes
        bool replaced = false;
        if (use_shape_for_elimination && input.get_partial_shape().is_static() &&
            reshape->get_output_partial_shape(0).is_static() && input.get_shape() == reshape->get_output_shape(0)) {
            // in case if elimination is not allowed we still can eliminate all transposes except last one
            replaced = replace_output_update_name(reshape->output(0), input);
        }

        if (!replaced) {
            reshape->input(0).replace_source_output(input);
            copy_runtime_info(nodes, reshape);
            return false;  // because root node wasn't replaced
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_b, matcher_name);
    this->register_matcher(m, callback);
}
