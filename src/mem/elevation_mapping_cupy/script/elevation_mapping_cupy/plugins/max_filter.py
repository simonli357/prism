#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List

from .plugin_manager import PluginBase


class MaxFilter(PluginBase):
    """This is a filter to fill in invalid cells with maximum values around.

    ...

    Attributes
    ----------
    width: int
        width of the elevation map.
    height: int
        height of the elevation map.
    dilation_size: int
        The size of the patch to search for maximum value for each iteration.
    iteration_n: int
        The number of iteration to repeat the same filter.
    """

    def __init__(self, cell_n: int = 100, dilation_size: int = 5, iteration_n: int = 5, **kwargs):
        super().__init__()
        self.iteration_n = iteration_n
        self.width = cell_n
        self.height = cell_n
        self.max_filtered = cp.zeros((self.width, self.height))
        self.max_filtered_mask = cp.zeros((self.width, self.height))
        self.max_filter_kernel = cp.ElementwiseKernel(
            in_params="raw U map, raw U mask",
            out_params="raw U newmap, raw U newmask",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                    const int layer = ${width} * ${height};
                    const int relative_idx = idx + ${width} * dy + dx;
                    return layer * layer_n + relative_idx;
                }
                __device__ bool is_inside(int idx) {
                    int idx_x = idx / ${width};
                    int idx_y = idx % ${width};
                    if (idx_x <= 0 || idx_x >= ${width} - 1) {
                        return false;
                    }
                    if (idx_y <= 0 || idx_y >= ${height} - 1) {
                        return false;
                    }
                    return true;
                }
                """
            ).substitute(width=self.width, height=self.height),
            operation=string.Template(
                """
                U valid = mask[get_map_idx(i, 0)];
                if (valid < 0.5) {
                    U max_value = -1000000.0;
                    for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                        for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                            int idx = get_relative_map_idx(i, dx, dy, 0);
                            if (!is_inside(idx)) {continue;}
                            U valid = mask[idx];
                            U value = map[idx];
                            if(valid > 0.5 && value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                    if (max_value > -1000000 + 1) {
                        newmap[get_map_idx(i, 0)] = max_value;
                        newmask[get_map_idx(i, 0)] = 0.6;
                    }
                }
                """
            ).substitute(dilation_size=dilation_size),
            name="max_filter_kernel",
        )

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
        
    ) -> cp.ndarray:
        self.max_filtered = elevation_map[0].copy()
        self.max_filtered_mask = elevation_map[2].copy()
        for i in range(self.iteration_n):
            self.max_filter_kernel(
                self.max_filtered.copy(),
                self.max_filtered_mask.copy(),
                self.max_filtered,
                self.max_filtered_mask,
                size=(self.width * self.height),
            )
            # If there's no more mask, break
            if (self.max_filtered_mask > 0.5).all():
                break
        max_filtered = cp.where(self.max_filtered_mask > 0.5, self.max_filtered.copy(), cp.nan)
        return max_filtered
