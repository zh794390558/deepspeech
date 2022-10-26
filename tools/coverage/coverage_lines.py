#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
usage: coverage_lines.py info_file expected
"""
import os
import sys


def get_lines(info_file):
    """
    Args:
        info_file (str): File generated by lcov.

    Returns:
        float: Coverage rate.
    """
    hits = 0.0
    total = 0.0

    with open(info_file) as info_file:
        for line in info_file:
            line = line.strip()

            if not line.startswith('DA:'):
                continue

            line = line[3:]

            total += 1

            if int(line.split(',')[1]) > 0:
                hits += 1

    if total == 0:
        print('no data found')
        exit()

    return hits / total


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit()

    info_file = sys.argv[1]
    expected = float(sys.argv[2])

    if not os.path.isfile(info_file):
        print('info file {} is not exists, ignored'.format(info_file))
        exit()

    actual = get_lines(info_file)
    actual = round(actual, 3)

    if actual < expected:
        print(
            'expected >= {} %, actual {} %, failed'.format(
                round(expected * 100, 1), round(actual * 100, 1)
            )
        )

        exit(1)

    print(
        'expected >= {} %, actual {} %, passed'.format(
            round(expected * 100, 1), round(actual * 100, 1)
        )
    )
