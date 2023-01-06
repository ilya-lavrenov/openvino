// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>
#include "memory_desc/dnnl_blocked_memory_desc.h"

#include "executors/matmul_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class MatMul : public Node {
public:
    MatMul(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool canFuse(const NodePtr& node) const override;
    bool created() const override;
    size_t getMaxBatch() const override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() - 1;
    }

    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr initPrimitiveAttr(const VectorDims& dims);

private:
    void setPostOps(dnnl::primitive_attr &attr, const VectorDims& dims, bool initWeights);

    std::vector<InferenceEngine::Precision> inputPrecisions;
    std::vector<InferenceEngine::Precision> outputPrecisions;

    std::string errorPrefix;
    MatMulAttrs matmulAttrs;
    std::shared_ptr<MatMulExecutor> execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
