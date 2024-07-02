"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import json


class QAContext:
    def __init__(self, user_context):
        self.user_context = user_context
        self.results = []

    def add_results(self, key, results):
        self.results.append({"key": key, "results": results})

    def to_dict(self):
        return {"user_context": self.user_context.to_dict(), "results": self.results}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
