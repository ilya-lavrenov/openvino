// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "mvn.hpp"
#include "x64/jit_mvn.hpp"
#include "common/ref_mvn.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct MVNExecutorDesc {
    impl_desc_type implType;
    MVNExecutorBuilderCPtr builder;
};

const std::vector<MVNExecutorDesc>& getMVNExecutorsList();

class MVNExecutorFactory : public ExecutorFactory {
public:
    MVNExecutorFactory(const MVNAttrs& mvnAttrs,
                       const std::vector<MemoryDescCPtr>& srcDescs,
                       const std::vector<MemoryDescCPtr>& dstDescs) : ExecutorFactory() {
        for (auto& desc : getMVNExecutorsList()) {
            if (desc.builder->isSupported(mvnAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~MVNExecutorFactory() = default;
    virtual MVNExecutorPtr makeExecutor(const MVNAttrs& mvnAttrs,
                                        const std::vector<MemoryDescCPtr>& srcDescs,
                                        const std::vector<MemoryDescCPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const MVNExecutorDesc* desc) {
            switch (desc->implType) {
#if defined(OV_CPU_X64)
                case impl_desc_type::jit_uni: {
                    auto builder = [&](const JitMVNExecutor::Key& key) -> MVNExecutorPtr {
                        auto executor = desc->builder->makeExecutor();
                        if (executor->init(mvnAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };

                    auto key = JitMVNExecutor::Key(mvnAttrs, srcDescs, dstDescs, attr);
                    auto res = runtimeCache->getOrCreate(key, builder);
                    return res.first;
                } break;
#endif
                default: {
                    auto executor = desc->builder->makeExecutor();
                    if (executor->init(mvnAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            MVNExecutorPtr ptr = nullptr;
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
    std::vector<MVNExecutorDesc> supportedDescs;
    const MVNExecutorDesc* chosenDesc = nullptr;
};

using MVNExecutorFactoryPtr = std::shared_ptr<MVNExecutorFactory>;
using MVNExecutorFactoryCPtr = std::shared_ptr<const MVNExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov