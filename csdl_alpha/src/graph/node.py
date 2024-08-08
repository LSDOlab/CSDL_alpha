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

        banned_paths = ['csdl_alpha/src', 'csdl_alpha/utils']
        # exception_paths = ['csdl_alpha/src/operations/loops']
        # exception_paths = ['test_']
        
        # banned_paths = ['/node', '/variable']
        exception_paths=[]
        
        if not any(path in info.filename for path in banned_paths):
            trace = [f"{info.filename}:{info.lineno} in {info.function}"]
            self.origin_info = {"filename": info.filename, "lineno": info.lineno, "function": info.function}
            set_origin = False

        while node_stack[-1].f_back is not None:
            node_stack.append(node_stack[-1].f_back)
            info = inspect.getframeinfo(node_stack[-1])
            if any(path in info.filename for path in exception_paths):
                pass
            elif any(path in info.filename for path in banned_paths):
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

    def info(self,) -> str:
        """returns a string containing information about the node

        Returns
        -------
        str
            information about the node
        """
        if self.name is not None:
            base_repr = f"{self.get_base_str()} ({self.name})"
        else:
            base_repr = f"{self.get_base_str()}"

        if self.trace is None or len(self.trace) == 0:
            return f"\'{base_repr}\'"
        else:
            return f"\'{base_repr} (from {self.trace[-1]})\'"
            
    def get_base_str(self) -> str:
        return f"{self.__class__.__name__} {hex(id(self))}"