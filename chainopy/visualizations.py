import xarray as xr
import matplotlib.pyplot as plt
import networkx as nx


def _visualize_tpm(transition_matrix, states):
    da_transition = xr.DataArray(
        transition_matrix,
        dims=["current_state", "next_state"],
        coords={"current_state": states, "next_state": states},
    )

    da_transition.plot.imshow(x="next_state", y="current_state", cmap="viridis")

    plt.show()


def _visualize_chain(transition_matrix, states):
    G = nx.DiGraph()

    for i, state in enumerate(states):
        G.add_node(state)
        for j, next_state in enumerate(states):
            probability = transition_matrix[i, j]
            if probability > 0:
                G.add_edge(state, next_state, weight=probability)

    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="lightblue",
        font_size=10,
        font_color="black",
        font_weight="bold",
        arrowsize=20,
    )

    edge_labels = {(i, j): f"{p:.2f}" for i, j, p in G.edges(data="weight")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()
