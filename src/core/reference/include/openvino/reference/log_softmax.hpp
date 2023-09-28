// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/shape_util.hpp"
#include "openvino/reference/reduce_max.hpp"
#include "openvino/reference/reduce_sum.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T>
void log_softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto temp_shape = ngraph::reduce(shape, axes, true);
    auto temp_elements = shape_size(temp_shape);
    auto temp_max = std::vector<T>(temp_elements, 0);
    auto temp_sum = std::vector<T>(temp_elements, 0);

    reduce_max(arg, temp_max.data(), shape, axes);

    CoordinateTransform transform(shape);
    CoordinateTransform temp_transform(temp_shape);
    for (const Coordinate& coord : transform) {
        Coordinate temp_coord = ngraph::reduce(coord, axes, true);
        out[transform.index(coord)] =
            static_cast<T>(std::exp(arg[transform.index(coord)] - temp_max[temp_transform.index(temp_coord)]));
    }

    reduce_sum(out, temp_sum.data(), shape, axes);

    for (const Coordinate& coord : transform) {
        Coordinate temp_coord = ngraph::reduce(coord, axes, true);
        out[transform.index(coord)] =
            static_cast<T>((arg[transform.index(coord)] - temp_max[temp_transform.index(temp_coord)]) -
                           std::log(temp_sum[temp_transform.index(temp_coord)]));
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace reference
}  // namespace ov
