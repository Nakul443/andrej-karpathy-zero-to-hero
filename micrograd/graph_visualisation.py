from graphviz import Digraph


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # format node label with optional variable name (label) and gradient (grad)
        label_parts = []
        if hasattr(n, "label") and n.label:
            label_parts.append(str(n.label))
        label_parts.append(f"data {n.data:.4f}")
        if hasattr(n, "grad"):
            label_parts.append(f"grad {n.grad:.4f}")
        dot.node(name=uid, label=f"{{ {' | '.join(label_parts)} }}", shape="record")

        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


if __name__ == "__main__":
    import importlib.util
    from pathlib import Path

    curr_dir = Path(__file__).parent
    source_file = curr_dir / "defining_data_structures_objects_part2.py"

    spec = importlib.util.spec_from_file_location(
        "defining_data_structures_objects_part2", source_file
    )
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Render to SVG
        output_path = curr_dir / "computational_graph"
        dot = draw_dot(mod.L)
        dot.render(str(output_path), format="svg", cleanup=True)
        print(f"Graph generated at: {output_path}.svg")
