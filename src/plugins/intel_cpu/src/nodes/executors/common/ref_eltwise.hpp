// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"

namespace ov {
namespace intel_cpu {

class RefEltwiseExecutor : public EltwiseExecutor {
public:
    RefEltwiseExecutor();

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    EltwiseAttrs eltwiseAttrs;
    impl_desc_type implType = impl_desc_type::ref;

    VectorDims _dims;
    std::vector<VectorDims> _src_offsets;
    VectorDims _dst_offsets;
    size_t _fullWorkAmount = 0;
    size_t _inputNum = 0;
    size_t _batchDimIdx = 0;
};

class RefEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    EltwiseExecutorPtr makeExecutor() const override {
        return std::make_shared<RefEltwiseExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov