from streamlit_agraph import agraph, Node, Edge, Config


def setup_graph(data):
    nodes = []
    edges = []
    for n in data["nodes"]:
        nodes.append(
            Node(id=n["~id"], label=n["~labels"][0], group=n["~labels"][0])
        )  # includes **kwargs
    for n in data["edges"]:
        edges.append(Edge(source=n["~start"], target=n["~end"], label=n["~type"]))

    config = Config(
        width=950,
        height=950,
        directed=True,
        physics=True,
        hierarchical=False,
    )

    return agraph(nodes=nodes, edges=edges, config=config)
