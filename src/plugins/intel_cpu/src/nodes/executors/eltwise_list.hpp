// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "eltwise.hpp"
#include "x64/jit_eltwise.hpp"
#include "common/ref_eltwise.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct EltwiseExecutorDesc {
    impl_desc_type implType;
    EltwiseExecutorBuilderCPtr builder;
};

const std::vector<EltwiseExecutorDesc>& getEltwiseExecutorsList();

class EltwiseExecutorFactory : public ExecutorFactory {
public:
    EltwiseExecutorFactory(const EltwiseAttrs& eltwiseAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs) : ExecutorFactory() {
        for (auto& desc : getEltwiseExecutorsList()) {
            if (desc.builder->isSupported(eltwiseAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~EltwiseExecutorFactory() = default;
    virtual EltwiseExecutorPtr makeExecutor(const EltwiseAttrs& eltwiseAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const std::vector<EltwisePostOp>& postOps) {
        auto build = [&](const EltwiseExecutorDesc* desc) {
            switch (desc->implType) {
                // case impl_desc_type::jit_uni: {
                //     auto builder = [&](const JitEltwiseExecutor::Key& key) -> EltwiseExecutorPtr {
                //         auto executor = desc->builder->makeExecutor();
                //         if (executor->init(eltwiseAttrs, srcDescs, dstDescs, attr)) {
                //             return executor;
                //         } else {
                //             return nullptr;
                //         }
                //     };

                //     auto key = JitEltwiseExecutor::Key(eltwiseAttrs, srcDescs, dstDescs, attr);
                //     auto res = runtimeCache->getOrCreate(key, builder);
                //     return res.first;
                // } break;
                default: {
                    auto executor = desc->builder->makeExecutor();
                    if (executor->init(eltwiseAttrs, srcDescs, dstDescs, postOps)) {
                        return executor;
                    }
                } break;
            }

            EltwiseExecutorPtr ptr = nullptr;
            return ptr;
        };


        if (chosenDesc) {
            if (auto executor = build(chosenDesc)) {
                return executor;
            }
        }

        for (const auto& sd : supportedDescs) {
            if (auto executor = build(&sd)) {
                chosenDesc = &sd;
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

private:
    std::vector<EltwiseExecutorDesc> supportedDescs;
    const EltwiseExecutorDesc* chosenDesc = nullptr;
};

using EltwiseExecutorFactoryPtr = std::shared_ptr<EltwiseExecutorFactory>;
using EltwiseExecutorFactoryCPtr = std::shared_ptr<const EltwiseExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov