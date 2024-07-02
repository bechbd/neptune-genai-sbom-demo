"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import boto3
import json
from botocore.config import Config


class NeptuneClient:
    def __init__(self, graph_id):
        self.graph_id = graph_id
        self.neptune = boto3.client(
            "neptune-graph",
            config=(
                Config(
                    retries={"total_max_attempts": 1, "mode": "standard"},
                    read_timeout=600,
                )
            ),
        )

    def execute_query(self, cypher, parameters={}):
        response = self.neptune.execute_query(
            graphIdentifier=self.graph_id,
            queryString=cypher,
            parameters=parameters,
            language="OPEN_CYPHER",
        )

        return json.loads(response["payload"].read())["results"]
