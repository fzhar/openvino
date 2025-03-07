// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {

class MKLDNNExperimentalDetectronPriorGridGeneratorNode : public MKLDNNNode {
public:
    MKLDNNExperimentalDetectronPriorGridGeneratorNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    // Inputs:
    //      priors, shape [n, 4]
    //      [feature_map], shape [b, c, h, w]
    //      [im_data], shape [b, 3, im_h, im_w]
    // Outputs:
    //      priors_grid, shape [m, 4]

    const int INPUT_PRIORS {0};
    const int INPUT_FEATUREMAP {1};
    const int INPUT_IMAGE {2};

    const int OUTPUT_ROIS {0};

    int grid_w_;
    int grid_h_;
    float stride_w_;
    float stride_h_;

    std::string errorPrefix;
};

}   // namespace intel_cpu
}   // namespace ov
