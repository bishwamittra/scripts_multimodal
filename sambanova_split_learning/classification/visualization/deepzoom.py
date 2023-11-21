#!/usr/bin/env python
#
# deepzoom_tile - Convert whole-slide images to Deep Zoom format
#
# Copyright (c) 2010-2015 Carnegie Mellon University
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
"""An example program to generate a Deep Zoom directory tree from a slide."""

from __future__ import print_function

import json
import os
import re
import shutil
import sys
from optparse import OptionParser
from unicodedata import normalize

import openslide
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""
    def __init__(self, slidepath, tile_size, overlap, offset, limit_bounds, desired_magnification, downsample_level):
        self._slidepath = slidepath
        self._slide = open_slide(slidepath)

        self._tile_size = tile_size
        self._overlap = overlap
        self._offset = offset
        self._limit_bounds = limit_bounds
        self._dzi_data = {}
        self.actual_tile_size = tile_size + 2 * overlap
        self.desired_magnification = desired_magnification

        # Will set self.dz and self.level
        self.dz, self.level = self.create_dz(power=self.get_best_power_of_2())
        self.cols, self.rows = self.dz.level_tiles[self.level]

        # Hard coded. Can change later
        self.downsample_level = downsample_level

        self.original_dimension = self.dz.level_dimensions[self.level]
        self.downsample_dimension = self.dz.level_dimensions[self.downsample_level]

        # Hard coded.
        # self.level = 16
        print(
            f"level: {self.level}, power: {self.get_best_power_of_2()}, mag: {self._slide.properties['openslide.objective-power']}, level_count: {self.dz.level_count}"
        )

    # Best power of 2 can be 0 but never negative
    def create_dz(self, associated=None, power=-1):
        """Run a single image from self._slide."""
        dz = DeepZoomGenerator(self._slide, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        level = (dz.level_count - 1) - power if power > -1 else -1  # Minus 1 from count for zero indexing of level
        return dz, level

    def get_best_power_of_2(self):
        slide_properties = self._slide.properties
        if 'openslide.objective-power' not in slide_properties._keys():
            print(self._slidepath, 'has no objective power property!')
            return -1
        slide_magnification = float(slide_properties['openslide.objective-power'])
        if self.desired_magnification > slide_magnification:
            print(self._slidepath,
                  'Slide magnification of %s is above desired magnification' % str(slide_magnification))
            return -1

        # Compute minimal power of 2 dividing slide magnification into desired magnification
        best_power_of_2 = 0
        while True:
            if self.desired_magnification * (2**best_power_of_2) >= slide_magnification:
                break
            best_power_of_2 += 1
        return best_power_of_2

    def get_downsample_dimension(self):
        """Hard coded to level 12"""
        return self.dz.level_dimensions[self.downsample_level]

    def get_original_coordinates(self, address):
        dimensions = self.dz.get_tile_dimensions(self.level, address)

        # smaller tile size
        if dimensions[0] != self.actual_tile_size or dimensions[0] != self.actual_tile_size:
            return None

        tile_coordinates = self.dz.get_tile_coordinates(self.level, address)[0]

        # negative coordinates
        if tile_coordinates[0] < 0 or tile_coordinates[1] < 0:
            return None

        return tile_coordinates

    def get_downsampled_coordinates(self, address):
        col_factor = self.downsample_dimension[0] / self.original_dimension[0]
        row_factor = self.downsample_dimension[1] / self.original_dimension[1]

        start_offset = self._overlap / self._tile_size
        original_coordinates = (512 * (address[0] - start_offset), 512 * (address[1] - start_offset))
        if original_coordinates[0] < 0 or original_coordinates[1] < 0:
            return None

        start_col = original_coordinates[0] * col_factor
        start_row = original_coordinates[1] * row_factor

        end_col = start_col + self.actual_tile_size * col_factor
        end_row = start_row + self.actual_tile_size * row_factor

        return (start_col, start_row, end_col, end_row)
