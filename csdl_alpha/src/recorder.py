from csdl_alpha.src.graph.graph import Graph

class Recorder:
    """
    The Recorder class represents a recorder object that is used to record data.

    Attributes:
        manager: The manager object that manages the recorders.
        graph_tree: The tree structure representing the graph hierarchy.
        namespace_tree: The tree structure representing the namespace hierarchy.
        active_graph_node: The currently active graph node.
        active_namespace: The currently active namespace node.
        active_graph: The currently active graph.
        active_namespace: The currently active namespace.

    Methods:
        start: Activates the recorder.
        stop: Deactivates the recorder.
        _enter_namespace: Enters a new namespace.
        _exit_namespace: Exits the current namespace.
        _enter_subgraph: Enters a new subgraph.
        _exit_subgraph: Exits the current subgraph.
        _add_node: Adds a node to the active namespace and graph.
    """

    def __init__(self, inline: bool = False, debug: bool = False):
        """
        Initializes a new instance of the Recorder class.
        """
        from csdl_alpha.api import manager
        self.manager = manager
        self.inline = inline
        self.debug = debug

        self.namespace_tree = Namespace(None)
        self.graph_tree = Tree(Graph())

        self.active_graph_node = self.graph_tree

        self.active_graph = self.active_graph_node.value
        self.active_namespace = self.namespace_tree

        self.node_graph_map = {}

        manager.constructed_recorders.append(self)
        
    def start(self):
        """
        Activates the recorder.
        """
        self.manager.activate_recorder(self)

    def stop(self):
        """
        Deactivates the recorder.
        """
        self.manager.deactivate_recorder(self)

    def _enter_namespace(self, name: str):
        """
        Enters a new namespace.

        Args:
            name: The name of the namespace to enter.
        """
        if not isinstance(name, str):
            raise TypeError("Name of namespace is not a string")
        
        if name in self.active_namespace.child_names:
            raise Exception("Attempting to enter existing namespace")
        
        self.active_namespace.child_names.add(name)

        if self.active_namespace.name is None:
            prepend = name
        else:
            prepend = self.active_namespace.prepend + '.' + name

        self.active_namespace = self.active_namespace.add_child(name, prepend=prepend)

    def _exit_namespace(self):
        """
        Exits the current namespace.
        """
        self.active_namespace = self.active_namespace.parent
        self.active_namespace = self.active_namespace

    def _enter_subgraph(self):
        """
        Enters a new subgraph.
        """
        self.active_graph_node = self.active_graph_node.add_child(Graph())
        self.active_graph = self.active_graph_node.value

    def _exit_subgraph(self):
        """
        Exits the current subgraph.
        """
        self.active_graph_node = self.active_graph_node.parent
        self.active_graph = self.active_graph_node.value

    def _add_node(self, node):
        """
        Adds a node to the active namespace and graph.

        Args:
            node: The node to add.
        """
        self.active_namespace.nodes.append(node)
        node.namespace = self.active_namespace
        self.active_graph.add_node(node)
        self.node_graph_map[node] = [self.active_graph]
        
    def _set_namespace(self, node):
        """
        sets namespace of node.
        """
        self.active_namespace.nodes.append(node)
        node.namespace = self.active_namespace_node.value

    def _add_edge(self, node_from, node_to):
        """
        Adds an edge between two nodes in the active graph.

        Args:
            node_from: The source node.
            node_to: The target node.
        """
        if node_from not in self.active_graph.node_table:
            raise ValueError(f"Node {node_from} not in graph")
        if node_to not in self.active_graph.node_table:
            raise ValueError(f"Node {node_to} not in graph")
        self.active_graph.add_edge(node_from, node_to)

class Tree:
    """
    Represents a tree data structure.

    Attributes:
        value: The value stored in the tree node.
        parent: The parent node of the current node.
        children: The list of child nodes of the current node.
    """

    def __init__(self, value, parent=None):
        """
        Initializes a new instance of the Tree class.

        Args:
            value: The value stored in the tree node.
            parent: The parent node of the current node.
        """
        self.value = value
        self.children = []
        self.parent = parent

    def add_child(self, value):
        """
        Adds a child node to the current node.

        Args:
            value: The value to be stored in the child node.

        Returns:
            The newly created child node.
        """
        child = Tree(value, parent=self)
        self.children.append(child)
        return child

class Namespace(Tree):
    """
    Represents a namespace.

    Attributes:
        name: The name of the namespace.
        nodes: The list of nodes in the namespace.
        prepend: The string to prepend to the namespace name.
    """
    def __init__(self, name, nodes=[], prepend=None, parent=None):
        """
        Initializes a new instance of the Namespace class.

        Args:
            name: The name of the namespace.
            nodes: The list of nodes in the namespace.
            prepend: The string to prepend to the namespace name.
        """
        self.name = name
        self.nodes = nodes
        self.prepend = prepend
        if prepend is None:
            self.prepend = name
        self.children = []
        self.parent = parent
        self.child_names = set()

    def add_child(self, name, nodes=[], prepend=None):
        """
        Adds a child namespace to the current namespace.

        Args:
            name: The name of the child namespace.
            nodes: The list of nodes in the child namespace.
            prepend: The string to prepend to the child namespace name.

        Returns:
            The newly created child namespace.
        """
        child = Namespace(name, nodes, prepend, parent=self)
        self.children.append(child)
        return child
