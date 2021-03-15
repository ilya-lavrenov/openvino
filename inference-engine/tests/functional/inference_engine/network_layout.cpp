// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_layouts.h>

using namespace std;
using namespace InferenceEngine;

using PartialLayoutTests = ::testing::Test;

TEST_F(PartialLayoutTests, smoke_createEmpty) {
    ASSERT_NO_THROW(PartialLayout layout;);
}

TEST_F(PartialLayoutTests, smoke_createCustomNHWC) {
    PartialLayout layout("NHWC");
    ASSERT_EQ(Layout::NHWC, layout);
    SizeVector refOrder = { 0, 2, 3, 1 };
    ASSERT_EQ(refOrder, layout.getOrder());

    EXPECT_TRUE(layout.hasBatch());
    EXPECT_TRUE(layout.hasWidth());
    EXPECT_TRUE(layout.hasHeight());
    EXPECT_TRUE(layout.hasChannels());

    EXPECT_EQ(0, layout.batch());
    EXPECT_EQ(3, layout.channels());
    EXPECT_EQ(1, layout.height());
    EXPECT_EQ(2, layout.width());

    SizeVector dims = { 1, 224, 224, 3 };
    EXPECT_EQ(1, dims[layout.batch()]);
    EXPECT_EQ(3, dims[layout.channels()]);
    EXPECT_EQ(224, dims[layout.height()]);
    EXPECT_EQ(224, dims[layout.width()]);
}

TEST_F(PartialLayoutTests, smoke_createCustomNHWC_FullySpecialize) {
    PartialLayout layout(SizeVector{ 0u, 2u, 3u, 1u });
    ASSERT_NE(Layout::NHWC, layout);  // dimensions are not named yet
    EXPECT_NO_THROW(layout.setBatch(0));
    EXPECT_NO_THROW(layout.setChannels(3));
    EXPECT_NO_THROW(layout.setHeight(1));
    EXPECT_NO_THROW(layout.setWidth(2));
    ASSERT_EQ(Layout::NHWC, layout);

    EXPECT_NO_THROW(layout.setBatch(0));
    EXPECT_THROW(layout.setBatch(1), details::InferenceEngineException);
}

TEST_F(PartialLayoutTests, smoke_createCustom0231) {
    SizeVector order = { 0, 2, 3, 1 };
    PartialLayout layout(order);
    ASSERT_NE(Layout::NHWC, layout);  // dimensions are not named
    ASSERT_EQ(order, layout.getOrder());
    EXPECT_FALSE(layout.hasBatch());
}

TEST_F(PartialLayoutTests, smoke_createCustomScalar) {
    PartialLayout layout("SCALAR");
    ASSERT_EQ(Layout::SCALAR, layout);
    ASSERT_EQ(0, layout.getOrder().size());
}

TEST_F(PartialLayoutTests, smoke_createBlockedDirectly) {
    ASSERT_THROW(PartialLayout("BLOCKED"), details::InferenceEngineException);
}

TEST_F(PartialLayoutTests, smoke_createBlocked) {
    PartialLayout layout(SizeVector{0, 1, 2, 3, 4, 5, 6, 7});
    EXPECT_EQ(Layout::BLOCKED, layout);
}

using PartialLayoutParamTests = ::testing::TestWithParam<Layout>;

TEST_P(PartialLayoutParamTests, createFromLayout) {
    Layout layout = GetParam();

    PartialLayout nlayout;
    ASSERT_NO_THROW(nlayout = PartialLayout(layout););

    // convert back to InferenceEngine::Layout
    EXPECT_EQ(layout, nlayout);

    // order
    SizeVector order = nlayout.getOrder();
    BlockingDesc blockingDesc(SizeVector(order.size(), 0), layout);
    ASSERT_EQ(blockingDesc.getOrder(), order);

    // normalizing order
    SizeVector newOrder(order.size());
    for (size_t i = 0; i < newOrder.size(); ++i) {
        newOrder[i] = i;
    }

    SizeVector normOrder = nlayout.convertToOrder(newOrder);
    for (size_t i = 0; i < normOrder.size(); ++i) {
        ASSERT_EQ(newOrder[i], order[normOrder[i]]);
    }

    const int ndims = normOrder.size();

    // has named dimensions
    if (layout != Layout::ANY && layout != Layout::SCALAR) {
        std::stringstream stream;
        stream << layout;
        std::string layoutStr = stream.str();

        for (size_t i = 0; i < layoutStr.length(); ++i) {
            if (layoutStr[i] == 'N') {
                EXPECT_TRUE(nlayout.hasBatch());
                EXPECT_EQ(i, nlayout.batch());
            } else if (layoutStr[i] == 'C') {
                EXPECT_TRUE(nlayout.hasChannels());
                EXPECT_EQ(i, nlayout.channels());
            } else if (layoutStr[i] == 'D') {
                EXPECT_TRUE(nlayout.hasDepth());
                EXPECT_EQ(i, nlayout.depth());
            } else if (layoutStr[i] == 'H') {
                EXPECT_TRUE(nlayout.hasHeight());
                EXPECT_EQ(i, nlayout.height());
            } else if (layoutStr[i] == 'W') {
                EXPECT_TRUE(nlayout.hasWidth());
                EXPECT_EQ(i, nlayout.width());
            }
        }

        // compare BlockingDesc
        SizeVector dims(ndims, 1);
        EXPECT_EQ(BlockingDesc(dims, layout), BlockingDesc(dims, nlayout.getOrder()));
    } else {
        EXPECT_FALSE(nlayout.hasBatch());
        EXPECT_FALSE(nlayout.hasChannels());
        EXPECT_FALSE(nlayout.hasDepth());
        EXPECT_FALSE(nlayout.hasWidth());
        EXPECT_FALSE(nlayout.hasHeight());
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
    // Layout::OIHW,  //!< NDHWC layout for operation weights
    // Layout::GOIHW,  //!< NDHWC layout for operation weights
    // Layout::OIDHW,  //!< NDHWC layout for operation weights
    // Layout::GOIDHW,  //!< NDHWC layout for operation weights

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

    // Layout::BLOCKED,  //!< A blocked layout
};

INSTANTIATE_TEST_CASE_P(smoke_, PartialLayoutParamTests, testing::ValuesIn(allLayouts));
