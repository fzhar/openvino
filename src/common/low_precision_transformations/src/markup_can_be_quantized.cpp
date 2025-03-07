// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_can_be_quantized.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"

using namespace ngraph;

ngraph::pass::low_precision::MarkupCanBeQuantized::MarkupCanBeQuantized(const std::vector<ngraph::element::Type> defaultPrecisions)
    : defaultPrecisions(defaultPrecisions) {}

bool ngraph::pass::low_precision::MarkupCanBeQuantized::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    auto setEmptyPrecisions = [](const std::shared_ptr<ngraph::Node>& node) {
        for (auto& input : node->inputs()) {
            auto& rt = input.get_rt_info();
            rt.emplace(
                    PrecisionsAttribute::get_type_info_static(),
                    PrecisionsAttribute(std::vector<element::Type>()));
        }
    };

    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0 || transformation_callback(node)) {
            continue;
        }

        if (const auto convolution = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node)) {
            if (!ConvolutionTransformation::isQuantizedStatic(convolution, defaultPrecisions)) {
                setEmptyPrecisions(convolution);
            }
            continue;
        }
        if (const auto convolutionBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
            if (!ConvolutionBackpropDataTransformation::isQuantizedStatic(convolutionBackpropData, defaultPrecisions)) {
                setEmptyPrecisions(convolutionBackpropData);
            }
            continue;
        }
        if (const auto groupConvolution = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node)) {
            if (!GroupConvolutionTransformation::isQuantizedStatic(groupConvolution, defaultPrecisions)) {
                setEmptyPrecisions(groupConvolution);
            }
            continue;
        }
        if (const auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(node)) {
            if (!ConcatTransformation::isQuantizedStatic(concat)) {
                setEmptyPrecisions(concat);
            }
            continue;
        }
    }
    return true;
}
