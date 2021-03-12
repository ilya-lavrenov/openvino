// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_layouts.h>

using namespace std;
using namespace InferenceEngine;

using NetworkLayoutTests = ::testing::Test;

static bool cannotBeParsed(Layout l) {
    return l == OIHW || l == GOIHW || l == OIDHW || l == GOIDHW ||
           l == HW || l == BLOCKED;
}

TEST_F(NetworkLayoutTests, smoke_createEmpty) {
    ASSERT_NO_THROW(NetworkLayout layout;);
}

TEST_F(NetworkLayoutTests, smoke_createCustomNHWC) {
    NetworkLayout layout(SizeVector{0, 2, 3, 1});
    layout.setDimensionIndexByName(NetworkLayout::BATCH, 0);
    layout.setDimensionIndexByName(NetworkLayout::HEIGHT, 1);
    layout.setDimensionIndexByName(NetworkLayout::WIDTH, 2);
    layout.setDimensionIndexByName(NetworkLayout::CHANNEL, 3);

    ASSERT_EQ(Layout::NHWC, layout);
}

TEST_F(NetworkLayoutTests, smoke_createCustomNCHW) {
    NetworkLayout layout(SizeVector{0, 1, 2, 3});
    layout.setDimensionIndexByName(NetworkLayout::BATCH, 0);
    layout.setDimensionIndexByName(NetworkLayout::CHANNEL, 1);
    layout.setDimensionIndexByName(NetworkLayout::HEIGHT, 2);
    layout.setDimensionIndexByName(NetworkLayout::WIDTH, 3);

    ASSERT_EQ(Layout::NCHW, layout);
}

TEST_F(NetworkLayoutTests, smoke_createCustomHWC) {
    NetworkLayout layout(SizeVector{1, 2, 0});
    layout.setDimensionIndexByName(NetworkLayout::HEIGHT, 0);
    layout.setDimensionIndexByName(NetworkLayout::WIDTH, 1);
    layout.setDimensionIndexByName(NetworkLayout::CHANNEL, 2);

    ASSERT_EQ(Layout::HWC, layout);
}

TEST_F(NetworkLayoutTests, smoke_createCustomUnknown) {
    NetworkLayout layout(SizeVector{0, 1, 2, 3});
    layout.setDimensionIndexByName(NetworkLayout::BATCH, 0);
    layout.setDimensionIndexByName(NetworkLayout::CHANNEL, 1);
    layout.setDimensionIndexByName(NetworkLayout::HEIGHT, 3);
    layout.setDimensionIndexByName(NetworkLayout::WIDTH, 2);

    ASSERT_EQ(Layout::BLOCKED, layout);
}

using NetworkLayoutParamTests = ::testing::TestWithParam<Layout>;

TEST_P(NetworkLayoutParamTests, createFromLayout) {
    Layout layout = GetParam();

    NetworkLayout nlayout;
    ASSERT_NO_THROW(nlayout = NetworkLayout(layout););

    // initialized?
    bool isInitialized = layout != Layout::ANY && layout != Layout::BLOCKED;
    ASSERT_EQ(isInitialized, nlayout.isInitialized());

    // convert back to InferenceEngine::Layout
    if (!cannotBeParsed(layout)) {
        ASSERT_EQ(layout, nlayout);
    }

    // order
    SizeVector order = nlayout.getOrder();
    BlockingDesc blockingDesc(SizeVector(nlayout.rank(), 0), layout);
    ASSERT_EQ(blockingDesc.getOrder(), order);

    // normalizing order
    SizeVector normOrder = nlayout.getNormalizingOrder();
    for (size_t i = 0; i < nlayout.rank(); ++i) {
        ASSERT_EQ(i, order[normOrder[i]]);
    }
}

static Layout allLayouts[] = {
    Layout::ANY,  //!< "any" layout

    // I/O data layouts
    Layout::NCHW,  //!< NCHW layout for input / output blobs
    Layout::NHWC,  //!< NHWC layout for input / output blobs
    Layout::NCDHW,  //!< NCDHW layout for input / output blobs
    Layout::NDHWC,  //!< NDHWC layout for input / output blobs

    // weight layouts
    Layout::OIHW,  //!< NDHWC layout for operation weights
    Layout::GOIHW,  //!< NDHWC layout for operation weights
    Layout::OIDHW,  //!< NDHWC layout for operation weights
    Layout::GOIDHW,  //!< NDHWC layout for operation weights

    // Scalar
    Layout::SCALAR,  //!< A scalar layout

    // bias layouts
    Layout::C,  //!< A bias layout for operation

    // Single image layouts
    Layout::CHW,  //!< A single image layout (e.g. for mean image)
    Layout::HWC,  //!< A single image layout (e.g. for mean image)

    // 2D
    Layout::HW,  //!< HW 2D layout
    Layout::NC,  //!< HC 2D layout
    Layout::CN,  //!< CN 2D layout

    Layout::BLOCKED,  //!< A blocked layout
};

INSTANTIATE_TEST_CASE_P(smoke_, NetworkLayoutParamTests, testing::ValuesIn(allLayouts));
