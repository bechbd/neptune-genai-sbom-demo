"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

from pipe import Pipe


def sink():
    def _sink(generator):
        for item in generator:
            pass

    return Pipe(_sink)
