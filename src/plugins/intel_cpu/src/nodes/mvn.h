// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

#include "executors/mvn_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class MVN : public Node {
public:
    MVN(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    inline bool getAcrossChannels() const {
        return mvnAttrs.initAcrossChannels_;
    }

    inline bool getNormalizeVariance() const {
        return mvnAttrs.normalizeVariance_;
    }

    bool canFuse(const NodePtr& node) const override;
    void prepareParams() override;


private:
    void setPostOps(dnnl::primitive_attr &attr, bool initWeights = false);

    void transformTo5DCase(const InferenceEngine::SizeVector& shape);

    std::vector<const void*> postOpsDataPtrs;

    MVNAttrs mvnAttrs;

    std::shared_ptr<MVNExecutor> execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
