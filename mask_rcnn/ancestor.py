import re


def ancestor(tensor, name, checked=None):
    """Finds the ancestor of a TF tensor in the computation graph.
    tensor: TensorFlow symbolic tensor.
    name: Name of ancestor tensor to find
    checked: For internal use. A list of tensors that were already
             searched to avoid loops in traversing the graph.
    """
    checked = checked if checked is not None else []
    # Put a limit on how deep we go to avoid very long loops
    if len(checked) > 500:
        return None
    # Convert name to a regex and allow matching a number prefix
    # because Keras adds them automatically
    if isinstance(name, str):
        name = re.compile(name.replace("/", r"(\_\d+)*/"))

    parents = tensor.op.inputs
    for p in parents:
        if p in checked:
            continue
        if bool(re.fullmatch(name, p.name)):
            return p
        checked.append(p)
        a = ancestor(p, name, checked)
        if a is not None:
            return a
    return None
