// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/ov_tensor_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
using ScatterNDUpdateShapes = std::vector<InputShape>;
using IndicesValues = std::vector<std::int64_t>;

struct ScatterNDUpdateLayerParams {
    ScatterNDUpdateShapes inputShapes;
    IndicesValues indicesValues;
};

using scatterUpdateParams = std::tuple<
    ScatterNDUpdateLayerParams,
    ElementType,        // input precision
    ElementType>;       // indices precision

class ScatterNDUpdateLayerCPUTest : public testing::WithParamInterface<scatterUpdateParams>, public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<scatterUpdateParams> obj) {
        ScatterNDUpdateLayerParams scatterParams;
        ElementType inputPrecision;
        ElementType idxPrecision;
        std::tie(scatterParams, inputPrecision, idxPrecision) = obj.param;
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesValues = scatterParams.indicesValues;

        std::ostringstream result;
        result << inputPrecision << "_IS=";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            for (const auto& targetShape : shape.second) {
                result << CommonTestUtils::vec2str(targetShape) << "_";
            }
            result << ")_";
        }
        result << "indices_values=" << CommonTestUtils::vec2str(indicesValues) << "_idx_precision=" << idxPrecision;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto& inputPrecision = funcInput.get_element_type();
            const auto& targetShape = targetInputStaticShapes[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor{ inputPrecision, targetShape };
                const auto indicesVals = std::get<0>(this->GetParam()).indicesValues;
                if (inputPrecision == ElementType::i32) {
                    auto data = tensor.data<std::int32_t>();
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        data[i] = static_cast<std::int32_t>(indicesVals[i]);
                    }
                } else if (inputPrecision == ElementType::i64) {
                    auto data = tensor.data<std::int64_t>();
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        data[i] = indicesVals[i];
                    }
                } else {
                    IE_THROW() << "GatherNDUpdate. Unsupported indices precision: " << inputPrecision;
                }
            } else {
                if (inputPrecision.is_real()) {
                    tensor = ov::test::utils::create_and_fill_tensor(inputPrecision, targetShape, 10, 0, 1000);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(inputPrecision, targetShape);
                }
            }
            inputs.insert({ funcInput.get_node_shared_ptr(), tensor });
        }
    }

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        ScatterNDUpdateLayerParams scatterParams;
        ElementType inputPrecision;
        ElementType idxPrecision;
        std::tie(scatterParams, inputPrecision, idxPrecision) = this->GetParam();
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesValues = scatterParams.indicesValues;

        init_input_shapes(inputShapes);
        selectedType = makeSelectedTypeStr("unknown", inputPrecision);

        auto dataParams = ngraph::builder::makeDynamicParams(inputPrecision, { inputDynamicShapes[0], inputDynamicShapes[2] });
        auto indicesParam = ngraph::builder::makeDynamicParams(idxPrecision, { inputDynamicShapes[1] });
        dataParams[0]->set_friendly_name("Param_1");
        indicesParam[0]->set_friendly_name("Param_2");
        dataParams[1]->set_friendly_name("Param_3");

        auto scatter = std::make_shared<ngraph::opset4::ScatterNDUpdate>(dataParams[0], indicesParam[0], dataParams[1]);

        ngraph::ParameterVector allParams{ dataParams[0], indicesParam[0], dataParams[1] };
        function = makeNgraphFunction(inputPrecision, allParams, scatter, "ScatterNDUpdateLayerCPUTest");
    }
};

TEST_P(ScatterNDUpdateLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(compiledModel, "ScatterUpdate");
}

const std::vector<ScatterNDUpdateLayerParams> scatterParams = {
    ScatterNDUpdateLayerParams{
        ScatterNDUpdateShapes{
            {{-1, -1, -1, -1, -1}, {{10, 9, 10, 9, 10}, {10, 1, 11, 2, 5}, {10, 15, 8, 1, 7}}},
            {{2, 2, 1}, {{2, 2, 1}, {2, 2, 1}, {2, 2, 1}}},
            {{-1, -1, -1, -1, -1, -1}, {{2, 2, 9, 10, 9, 10}, {2, 2, 1, 11, 2, 5}, {2, 2, 15, 8, 1, 7}}},
        },
        IndicesValues{ 5, 6, 2, 8 }
    },
    ScatterNDUpdateLayerParams{
        ScatterNDUpdateShapes{
            {{-1, -1, -1, -1}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, 3}, {{2, 3}, {2, 3}, {2, 3}}},
            {{-1, -1}, {{2, 11}, {2, 12}, {2, 8}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 }
    },
    ScatterNDUpdateLayerParams{
        ScatterNDUpdateShapes{
            {{{3, 10}, -1, {3, 9}, -1}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, 3}, {{2, 3}, {2, 3}, {2, 3}}},
            {{{2, 4}, -1}, {{2, 11}, {2, 12}, {2, 8}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 }
    },
    ScatterNDUpdateLayerParams{
        ScatterNDUpdateShapes{
            {{{3, 10}, {4, 11}, {3, 9}, {8, 15}}, {{ 10, 9, 9, 11 }, { 7, 5, 3, 12 }, { 3, 4, 9, 8 }}},
            {{2, 3}, {{2, 3}, {2, 3}, {2, 3}}},
            {{{2, 4}, -1}, {{2, 11}, {2, 12}, {2, 8}}}
        },
        IndicesValues{ 0, 1, 1, 2, 2, 2 }
    },
};

const std::vector<ElementType> inputPrecisions = {
    ElementType::f32,
    ElementType::i32,
};

const std::vector<ElementType> constantPrecisions = {
    ElementType::i32,
    ElementType::i64,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ScatterNDUpdateLayerCPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterParams),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterNDUpdateLayerCPUTest::getTestCaseName);
} // namespace CPULayerTestsDefinitions
