from dataclasses import dataclass
from csdl_alpha.src.recorder import Namespace
import inspect

class Node(object):
    """
    Represents a node in a graph.

    Attributes:
        namespace (Namespace): The namespace of the node.
    """
    namespace: Namespace = None

    def __init__(self) -> None:
        # recorder object
        import csdl_alpha
        self.recorder = csdl_alpha.get_current_recorder()
        self.recorder._set_namespace(self)
        self.trace = None

        if self.recorder.debug:
            self._apply_debug()

    def _apply_debug(self):
        """
        Adds useful debugging information to a node.

        Args:
            node: The node to apply debugging to.
        """

        from csdl_alpha.src.graph.node import Node
        node_stack = [inspect.currentframe().f_back.f_back.f_back]
        info = inspect.getframeinfo(node_stack[0])
        trace = [f"{info.filename}:{info.lineno}"]
        while node_stack[0].f_back is not None:
            node_stack.insert(0, node_stack[0].f_back)
            info = inspect.getframeinfo(node_stack[0])
            trace.insert(0, f"{info.filename}:{info.lineno}")
        self.trace = trace

    def print_trace(self):
        """
        Prints the trace of the node.
        """
        if self.trace is None:
            print("No trace available.")
            return
        for item in self.trace:
            print(item)

    # def __eq__(self, other):
    #     return self is other
    # def __hash__(self):
    #     return id(self)