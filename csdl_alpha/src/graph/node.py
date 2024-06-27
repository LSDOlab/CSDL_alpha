from dataclasses import dataclass
import inspect

class Node(object):
    """
    Represents a node in a graph.

    Attributes:
        namespace (Namespace): The namespace of the node.
    """
    # __slots__ = "recorder", "namespace", "trace", "hierarchy", "is_input", "is_implicit", "save", "names", "name", "value", "shape", "size", "tags"

    def __init__(self) -> None:
        from csdl_alpha.src.recorder import Namespace
        self.namespace: Namespace = None
        # recorder object
        import csdl_alpha
        self.recorder = csdl_alpha.get_current_recorder()
        self.recorder._set_namespace(self)
        self.trace = None
        self.origin_info = None

        if self.recorder.debug:
            self._apply_debug()

    def _apply_debug(self):
        """
        Adds useful debugging information to a node.

        Args:
            node: The node to apply debugging to.
        """
        # TODO: have internal frame removal be an option (on recorder?)

        from csdl_alpha.src.graph.variable import Variable
        from csdl_alpha.src.graph.operation import Operation
        node_stack = [inspect.currentframe()]
        set_origin = True
        info = inspect.getframeinfo(node_stack[-1])
        trace = []
        if 'csdl_alpha/src' not in info.filename and 'csdl_alpha/utils' not in info.filename:
            trace = [f"{info.filename}:{info.lineno} in {info.function}"]
            self.origin_info = {"filename": info.filename, "lineno": info.lineno, "function": info.function}
            set_origin = False

        while node_stack[-1].f_back is not None:
            node_stack.append(node_stack[-1].f_back)
            info = inspect.getframeinfo(node_stack[-1])
            if 'csdl_alpha/src' in info.filename or 'csdl_alpha/utils' in info.filename:
                continue
            if set_origin:
                self.origin_info = {"filename": info.filename, "lineno": info.lineno, "function": info.function}
                set_origin = False
            trace.append(f"{info.filename}:{info.lineno} in {info.function}")
        self.trace = list(reversed(trace))

    def print_trace(self, tab = False):
        """
        Prints the trace of the node.
        """
        if self.trace is None:
            print("No trace available.")
            return
        for item in self.trace:
            if tab:
                print('\t', end = '')
            print(item)

    # def __eq__(self, other):
    #     return self is other
    # def __hash__(self):
    #     return id(self)