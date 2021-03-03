// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/preprocessing/mean_image_or_value.hpp"
#include "transformations/preprocessing/std_scale.hpp"

#include "preprocessing.hpp"

class LayoutNormalization : public ngraph::pass::MatcherPass {
public:
    using LayoutMap = std::map<std::string, InferenceEngine::Layout>;
    NGRAPH_RTTI_DECLARATION;

    explicit LayoutNormalization(const LayoutMap& inputInfoMap) {
        auto param = ngraph::pattern::wrap_type<ngraph::opset3::Parameter>();

        ngraph::matcher_pass_callback callback = [=] (ngraph::pattern::Matcher& m) {
            auto param = std::dynamic_pointer_cast<ngraph::opset3::Parameter>(m.get_match_root());
            if (!param) {
                return false;
            }

            auto it = inputInfoMap.find(param->get_friendly_name());
            if (it == inputInfoMap.end() || it->second == InferenceEngine::Layout::ANY) {
                return false;
            }

            auto pShape = param->get_partial_shape();
            if (pShape.rank().is_dynamic())
                return false;

            InferenceEngine::SizeVector dummyDims(pShape.rank().get_length(), 0);
            auto ieLayout = InferenceEngine::TensorDesc::getLayoutByDims(dummyDims);

            if (it->second == ieLayout)
                return true;

            InferenceEngine::TensorDesc dummyDesc(InferenceEngine::Precision::U8, dummyDims, it->second);
            auto order = dummyDesc.getBlockingDesc().getOrder();
            ngraph::Shape newShape(dummyDims.size());

            for (size_t i = 0; i < dummyDims.size(); ++i) {
                for (size_t j = 0; j < dummyDims.size(); ++j) {
                    if (order[j] == i) {
                        newShape[i] = pShape[j].get_length();
                        break;
                    }
                }
                std::cout << order[i] << " " << newShape[i] << std::endl;
            }
            std::cout << std::endl;

            auto input_order = ngraph::opset3::Constant::create(
                ngraph::element::i32, ngraph::Shape{dummyDims.size()}, order);
            auto new_shape = ngraph::opset3::Constant::create(
                ngraph::element::i32, ngraph::Shape{dummyDims.size()}, pShape.get_shape());
            auto copy_param = std::make_shared<ngraph::opset3::Parameter>(
                param->get_element_type(), newShape);
            auto transpose = std::make_shared<ngraph::opset3::Transpose>(copy_param, input_order);
            auto reshape = std::make_shared<ngraph::opset3::Reshape>(transpose, new_shape, false);

            ngraph::replace_node(param, reshape);
            transpose->set_argument(0, param);
            param->set_partial_shape(newShape);

            // Return true as the root node was changed
            return true;
        };

        // Register pattern with Parameter operation as a pattern root node
        auto m = std::make_shared<ngraph::pattern::Matcher>(param, "LayoutNormalization");
        // Register Matcher
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(LayoutNormalization, "LayoutNormalization", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::AddPreprocessing, "AddPreprocessing", 0);

ngraph::pass::AddPreprocessing::AddPreprocessing(const InferenceEngine::InputsDataMap & inputInfoMap)
    : m_inputInfoMap(inputInfoMap) {
}

bool ngraph::pass::AddPreprocessing::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::AddMeanSubtract::MeanMap meanMap;
    ngraph::pass::AddStdScale::ScaleMap scaleMap;
    LayoutNormalization::LayoutMap layoutMap;

    for (const auto & it : m_inputInfoMap) {
        bool has_scales = false, has_mean_values = false, has_mean_image = false;
        const InferenceEngine::PreProcessInfo & pInfo = it.second->getPreProcess();
        const auto & inputDims = it.second->getTensorDesc().getDims();
        const size_t cn = pInfo.getNumberOfChannels();
        std::vector<float> meanValues(cn), stdScales(cn);
        InferenceEngine::Blob::Ptr meanImage = nullptr;

        for (size_t c = 0; c < cn; ++c) {
            if ((stdScales[c] = pInfo[c]->stdScale) != 1.0f) {
                has_scales = true;
            }

            if ((meanValues[c] = pInfo[c]->meanValue) != 0.0f) {
                has_mean_values = true;
            }

            if (pInfo[c]->meanData != nullptr) {
                has_mean_image = true;
                if (c == 0) {
                    meanImage = pInfo[c]->meanData;
                    NGRAPH_CHECK(meanImage->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32,
                        "Only InferenceEngine::Precision::FP32 precision is supported for PreProcessChannel::meanData");
                } else {
                    NGRAPH_CHECK(meanImage->getTensorDesc() == pInfo[c]->meanData->getTensorDesc(),
                        "TensorDesc for PreProcessChannel::meanData must be equal");
                }
            }
        }

        layoutMap[it.first] = it.second->getNetworkLayout();

        // no preprocessing for current input
        if (!has_mean_values && !has_scales && !has_mean_image) {
            continue;
        }

        NGRAPH_CHECK(!(has_mean_image && has_scales),
            "Only PreProcessChannel::meanData or PreProcessChannel::meanValue can be set.");

        if (has_scales) {
            ngraph::Shape shape(inputDims.size(), 1);
            shape[1] = stdScales.size(); // C
            scaleMap[it.first] = ngraph::opset3::Constant::create(ngraph::element::f32, shape, stdScales);
        }

        if (has_mean_values) {
            ngraph::Shape shape(inputDims.size(), 1);
            shape[1] = meanValues.size(); // C
            meanMap[it.first] = ngraph::opset3::Constant::create(ngraph::element::f32, shape, meanValues);
        } else if (has_mean_image) {
            ngraph::Shape shape = { cn };
            auto dims = meanImage->getTensorDesc().getDims();
            std::copy(dims.begin(), dims.end(), std::back_inserter(shape));

            std::vector<float> meanImageData(ngraph::shape_size(shape));
            for (size_t c = 0, i = 0; c < cn; ++c) {
                auto lm = pInfo[c]->meanData->buffer();
                const float *data = lm.as<const float *>();

                std::memcpy(&meanImageData[i], data, meanImage->byteSize());
                i += meanImage->size();
            }

            meanMap[it.first] = ngraph::opset3::Constant::create(ngraph::element::f32,
                shape, meanImageData);
        }
    }

    ngraph::pass::Manager manager(get_pass_config());
    auto preproc = manager.register_pass<ngraph::pass::GraphRewrite>();

    // if (!scaleMap.empty()) {
    //     preproc->add_matcher<ngraph::pass::AddStdScale>(scaleMap);
    // }
    // if (!meanMap.empty()) {
    //     preproc->add_matcher<ngraph::pass::AddMeanSubtract>(meanMap);
    // }

    preproc->add_matcher<LayoutNormalization>(layoutMap);

    manager.run_passes(f);

    return false;
}
