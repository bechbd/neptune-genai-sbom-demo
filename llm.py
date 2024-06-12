from langchain_community.graphs import NeptuneAnalyticsGraph
from langchain.chat_models import BedrockChat
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain.chains import NeptuneOpenCypherQAChain
import boto3
import json
from langchain.prompts import PromptTemplate
from enum import Enum
import os
import streamlit as st

vulnerability_list = None
QUERY_TYPES = Enum("QUERY_TYPES", ["KnowledgeGraph", "RAG", "GraphRAG", "Unknown"])

VULNERABILITY_VSS_QUERY = """
    MATCH (n:Vulnerability {id: $id})
    CALL neptune.algo.vectors.topKByNode(n)
    YIELD node, score
    RETURN node.id as id, node.description as description, score
    ORDER BY score ASC
"""

VULNERABILITY_GRAPH_RAG_QUERY_NODES = """
    MATCH (n:Vulnerability {id: $id})
    CALL neptune.algo.vectors.topKByNode(n)
    YIELD node, score
    WITH n, node ORDER BY score ASC
    MATCH p=(n)-[:AFFECTS]-(c:Component)-[:AFFECTS]-(noe:Vulnerability)
    WITH nodes(p) as nodes
    UNWIND nodes as n
    RETURN collect(distinct n) as nodes
"""

VULNERABILITY_GRAPH_RAG_QUERY_EDGES = """
    MATCH (n:Vulnerability {id: $id})
    CALL neptune.algo.vectors.topKByNode(n)
    YIELD node, score
    WITH n, node ORDER BY score ASC
    MATCH p=(n)-[:AFFECTS]-(c:Component)-[:AFFECTS]-(noe:Vulnerability)
    WITH relationships(p) as edges
    UNWIND edges as e
    RETURN collect(distinct e) as edges
"""

VULNERABILITY_LIST_QUERY = """
    MATCH (n:Vulnerability)
    RETURN n.id AS id ORDER BY id
"""

graph_id = os.getenv("GRAPH_ID")

llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=boto3.client("bedrock-runtime"),
    model_kwargs={"temperature": 0},
)
graph = NeptuneAnalyticsGraph(graph_identifier=graph_id)
neptune_client = boto3.client("neptune-graph")

chain = NeptuneOpenCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    return_direct=True,
    return_intermediate_steps=True,
    extra_instructions="""Wrap all property names in backticks exclude label names. 
                        All comparisons with string values should be done in lowercase
                        If you don't know how to write a query given the prompt return 'I don't know' """,
)

chat = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.1},
)


def determine_question_type(prompt) -> QUERY_TYPES:
    template = f"""
        You are an expert in determining if a question needs a document, a database query, or both.  Give the question below:
        {prompt}
        Is this best as a database query, a document query, or both?  Only answer database, document, both, or i don't know
            """
    response = chat.invoke(template)
    if response.content.lower() == "database":
        return QUERY_TYPES.KnowledgeGraph
    elif response.content.lower() == "document":
        return QUERY_TYPES.RAG
    elif response.content.lower() == "both":
        return QUERY_TYPES.GraphRAG
    else:
        return QUERY_TYPES.Unknown


def determine_query_information(prompt, query_type):
    template = f"""
    You are an expert in Software Bill Of Materials.  Given the question below:
    {prompt}
    Tell me if the user wants to know about a Document, Component, License, Vulnerability, or Reference element 
    and what value they would like to find out about.  
    The answer must be two lines, with the first line being the element type name and the second line being the id.  Do not add any additional words in the answer
                """
    response = chat.invoke(template)
    lines = response.content.split("\n")
    if len(lines) == 2:
        node_labels = graph._get_labels()
        label = None
        if lines[0] in node_labels[0]:
            label = lines[0]
        value_id = lines[1]

        match label:
            case "Vulnerability":
                if query_type == QUERY_TYPES.RAG:
                    resp = run_graph_query(VULNERABILITY_VSS_QUERY, {"id": value_id})
                elif query_type == QUERY_TYPES.GraphRAG:
                    resp = {"subgraph": {}}
                    res = run_graph_query(
                        VULNERABILITY_GRAPH_RAG_QUERY_NODES, {"id": value_id}
                    )
                    resp["subgraph"]["nodes"] = res[0]["nodes"]
                    res = run_graph_query(
                        VULNERABILITY_GRAPH_RAG_QUERY_EDGES, {"id": value_id}
                    )
                    resp["subgraph"]["edges"] = res[0]["edges"]

                else:
                    resp = "The information you requested is not currently supported by this application.  Please try again."
            case _:
                resp = "The information you requested is not currently supported by this application.  Please try again."
        return resp


def run_natural_language_query(prompt):
    resp = chain.invoke(prompt)
    return resp["result"]


@st.cache_data
def get_vulnerability_list():
    data = run_graph_query(VULNERABILITY_LIST_QUERY)
    return [d["id"] for d in data]


def run_rag_query(query):
    resp = determine_query_information(query, QUERY_TYPES.RAG)
    return resp


def run_graphrag_query(query):
    resp = determine_query_information(query, QUERY_TYPES.GraphRAG)
    return resp


def run_graph_query(query, parameters={}):
    resp = neptune_client.execute_query(
        graphIdentifier=graph_id,
        queryString=query,
        parameters=parameters,
        language="OPEN_CYPHER",
    )
    data = json.loads(resp["payload"].read().decode("UTF-8"))
    return data["results"]
