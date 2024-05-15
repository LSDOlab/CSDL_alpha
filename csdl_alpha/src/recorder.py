from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.utils.inputs import get_type_string
import inspect

class Recorder:
    """
    The Recorder class assembles CSDL variables and operations into a computational graph.

    Attributes
    ----------
    manager : Manager
        The global manager object that manages the recorders.
    active_graph_node : Tree
        The currently active graph node.
    active_namespace : Namespace 
        The currently active namespace node.
    active_graph : Graph
        The currently active graph.
    """

    def __init__(self, 
                 inline: bool = False, 
                 debug: bool = False, 
                 expand_ops: bool = False,
                 auto_hierarchy: bool = False):
        """
        Initializes a Recorder object.

        Parameters
        ----------
        inline : bool, optional
            Specifies whether to run inline evaluations, by default False.
        debug : bool, optional
            Specifies whether to enable debug mode, by default False.
        expand_ops : bool, optional
            Specifies whether to expand composed operations, by default False.
        auto_hierarchy : bool, optional
            Specifies whether to automatically create a hierarchy, by default False.
        """
        from csdl_alpha.api import manager
        self.manager = manager
        self.inline = inline
        self.debug = debug
        self.expand_ops = expand_ops
        self.auto_hierarchy = auto_hierarchy

        self.hierarchy = 0

        # keep track of inline stuff for loops
        self._in_loop = False
        self._reset_loops = False

        self.design_variables = {}
        self.constraints = {}
        self.objectives = {}

        self.namespace_tree = Namespace(None)
        self.active_namespace = self.namespace_tree

        # TODO: unbloat...
        self.graph_tree = Tree(Graph())
        self.active_graph:Graph = self.graph_tree.value
        self.node_graph_map = {}
        self.active_graph_stack = [self.active_graph]
        self.graph_to_tree_node_map = {
            self.active_graph: self.graph_tree
        }

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

    def execute(self):
        """
        Executes the current active graph inline and updates all variable values
        """
        #TODO: TEST TEST TEST TEST

        self.active_graph.execute_inline()

    def gather_insights(self)->dict:
        """
        UNTESTED!
        """
        #TODO: TEST TEST TEST TEST

        from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
        from csdl_alpha.src.graph.variable import Variable

        information_dict = {}
        information_dict['names2nodes'] = {}
        information_dict['nodes2graphs'] = {}
        information_dict['input_nodes'] = set()
        information_dict['graph_tree'] = {}
        information_dict['analytics'] = {
            'number of nodes': 0,
            'number of edges': 0,
            'number of variables': 0,
            'number of operations': 0,
            'number of namespaces': 0,
            'number of graphs': 0,
        }

        all_nodes = set()

        root_graph = self.get_root_graph()
        graphs_to_process = [root_graph]
        while len(graphs_to_process) > 0:
            # current graph
            current_graph = graphs_to_process.pop(0)
            current_num_ops = 0

            # initialize graph tree and analytics
            information_dict['graph_tree'][current_graph] = []
            information_dict['analytics']['number of graphs'] += 1

            # iterate over all nodes in the current graph and store information
            for node in current_graph.node_table:
                information_dict['analytics']['number of nodes'] += 1
                if isinstance(node, Variable):

                    # store information about the variable once
                    if node not in all_nodes:
                        for name in node.names:
                            information_dict['names2nodes'][name] = node
                        information_dict['analytics']['number of variables'] += 1

                    all_nodes.add(node)

                    # track all input nodes of the RECORDER (not all subgraphs)
                    if current_graph is root_graph:
                        if len(current_graph.predecessors(node)) == 0:
                            information_dict['input_nodes'].add(node)
                        
                else:
                    information_dict['analytics']['number of operations'] += 1
                    if isinstance(node, SubgraphOperation):
                        subgraph = node.get_subgraph()
                        graphs_to_process.append(subgraph)
                        information_dict['graph_tree'][current_graph].append((node, subgraph))
                    else:
                        current_num_ops += 1

                # track all graphs that the node is in
                if node not in information_dict['nodes2graphs']:
                    information_dict['nodes2graphs'][node] = [current_graph]
                else:
                    information_dict['nodes2graphs'][node].append(current_graph)
            information_dict['graph_tree'][current_graph].append((None, f'(+{current_num_ops} ops)'))

        return information_dict

    def print_graph_structure(self):
        """
        prints graph tree structure like:

        root
            graph1
                graph2
            graph3
                graph4
                graph5
        """
        #TODO: TEST TEST TEST TEST
        graph_tree = self.gather_insights()['graph_tree']

        # from https://stackoverflow.com/questions/51903172/how-to-display-a-tree-in-python-similar-to-msdos-tree-command 
        def ptree(parent, tree, indent=''):


            if isinstance(parent, str):
                print(parent)
            else:
                print(parent.name)

            if parent not in tree:
                return

            indent += ' '

            for child in tree[parent][:-1]:
                print(indent + '|' + '-' * 4, end='')
                ptree(child[1], tree, indent + '|' + ' ' * 4)

            if len(tree[parent]) > 0:
                child = tree[parent][-1]
                print(indent + '`' + '-' * 4, end='')
                ptree(child[1], tree, indent + ' ' * 4)

        ptree(self.get_root_graph(), graph_tree)

    def _enter_namespace(self, name: str):
        """
        Enters a new namespace.

        Arguments
        ---------
            name: The name of the namespace to enter.
        """
        if not isinstance(name, str):
            raise TypeError("Name of namespace is not a string")
        
        if name in self.active_namespace.child_names:
            raise Exception("Attempting to enter existing namespace")
        
        self.hierarchy += 1
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
        if self.active_namespace.parent is None:
            raise Exception("Attempting to exit root namespace")
        self.hierarchy -= 1
        self.active_namespace = self.active_namespace.parent
        self.active_namespace = self.active_namespace

    def _enter_subgraph(
            self,
            add_missing_variables: bool = False,
            name:str = None,
            graph:Graph = None,
        ):
        """
        Enters a new subgraph.
        """
        #TODO: TEST TEST TEST TEST

        if not isinstance(add_missing_variables, bool):
            raise TypeError(f"add_missing_variables must be a boolean. Got {get_type_string(add_missing_variables)}")
        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string. Got {get_type_string(name)}")
        if graph is not None and not isinstance(graph, Graph):
            raise TypeError(f"graph must be a Graph object. Got {get_type_string(graph)}")

        if graph is None:
            # Add new graph to tree
            active_graph_node = self.graph_to_tree_node_map[self.active_graph].add_child(Graph(name = name))
            self.active_graph = active_graph_node.value

            # Add new graph to graph to tree node map
            self.graph_to_tree_node_map[self.active_graph] = active_graph_node
            self.active_graph_stack.append(self.active_graph)
            self.active_graph.add_missing_variables = add_missing_variables
            self.active_graph.inputs = []
        else:
            self.active_graph = graph
            self.active_graph_stack.append(self.active_graph)

    def _exit_subgraph(self):
        """
        Exits the current subgraph.
        """
        self.active_graph_stack.pop()
        self.active_graph = self.active_graph_stack[-1]

        # self.active_graph_node = self.active_graph_node.parent
        # self.active_graph = self.active_graph_node.value

    def _add_node(self, node):
        """
        Adds a node to the active namespace and graph.

        Args:
            node: The node to add.
        """
        self.active_graph.add_node(node)
        self.node_graph_map[node] = [self.active_graph]

    def _set_namespace(self, node):
        """
        sets namespace of node.
        """
        from csdl_alpha.src.graph.variable import Variable
        self.active_namespace.nodes.append(node)
        node.namespace = self.active_namespace
        if self.auto_hierarchy and isinstance(node, Variable):
            node.set_hierarchy(self.hierarchy)

    def _add_edge(self, node_from, node_to):
        """
        Adds an edge between two nodes in the active graph.

        Args:
            node_from: The source node.
            node_to: The target node.
        """
        from csdl_alpha.src.graph.variable import Variable
        graph = self.active_graph
        if node_from not in graph.node_table: # TODO: consider changing node_graph_map to reflect this
            if graph.add_missing_variables and isinstance(node_from, Variable):
                graph.add_node(node_from)
                if not node_from in graph.inputs: graph.inputs.append(node_from)
            else:
                raise ValueError(f"Node {node_from.name} not in graph")
        if node_to not in graph.node_table:
            # if graph.add_missing_variables and isinstance(node_to, Variable):
            #     graph.add_node(node_to)
            # else:
            #     raise ValueError(f"Node {node_to} not in graph")
            raise ValueError(f"Node {node_to.name} not in graph")
        graph.add_edge(node_from, node_to)

    def _add_design_variable(self, variable, upper, lower, scalar):
        """
        Adds a design variable to the recorder.

        Args:
            variable: The design variable.
            upper: The upper bound of the design variable.
            lower: The lower bound of the design variable.
            scalar: The scalar value of the design variable.
        """
        self.design_variables[variable] = (upper, lower, scalar)

    def _add_constraint(self, variable, upper, lower, scalar):
        """
        Adds a constraint to the recorder.

        Args:
            variable: The constraint variable.
            upper: The upper bound of the constraint.
            lower: The lower bound of the constraint.
            scalar: The scalar value of the constraint.
        """
        self.constraints[variable] = (upper, lower, scalar)

    def _add_objective(self, variable, scalar):
        """
        Adds an objective to the recorder.

        Args:
            variable: The objective variable.
            scalar: The scalar value of the objective.
        """
        self.objectives[variable] = scalar

    def _delete_current_graph(self):
        """
        Deletes the current graph.
        """
        current_graph_node = self.active_graph_node
        parent_graph_node = current_graph_node.parent
        parent_graph_node.children.remove(current_graph_node)
        
        self.active_graph_node = parent_graph_node
        self.active_graph = parent_graph_node.value

    def visualize_graph(
            self,
            filename: str = 'image',
            visualize_style: str = 'namespace',
            trim_loops = False,
            format = 'svg',
            )->None:
        """
        Visualizes the graph.
        - 'namespace' visualizes the top-level graph where nodes are grouped by namespace
        - 'hierarchical' visualizes all graphs including subgraph operations for debugging. Always saves as a .svg file.

        Parameters
        ----------
        filename : str, optional
            The filename to save the visualization to, by default 'image'
        visualize_style : str, optional
            The style of visualization, by default 'namespace'
        trim_loops : bool, optional
            Whether to trim loops, by default False
        format : str, optional
            The format of the visualization, by default 'svg'

        """
        if visualize_style == 'namespace':
            self.active_graph.visualize(filename, trim_loops=trim_loops, format = format)
        elif visualize_style == 'hierarchical':
            self.visualize_hierarchical(filename)

        else:
            raise ValueError(f"Invalid visualize_style: {visualize_style}. Must be 'namespace' or 'hierarchical'")

    def visualize_hierarchical(self, filename):
        import pydot

        # Get the graph tree structure
        dot = pydot.Dot(graph_type='digraph')
        insights = self.gather_insights()

        # utility functions for naming nodes
        def get_raw_node_string(node):
            return str(node).split()[-1][:-1]

        def build_unique_node_name(node, parent):
            node_id = get_raw_node_string(node)
            parent_id = get_raw_node_string(parent)
            return f'{str(node_id)}_{str(parent_id)}'
        
        def name_single_node(node):
            return f'{get_raw_node_string(node)}_{node.name}'

        def name_node(node):
            from csdl_alpha.src.graph.variable import Variable
            attr_dict = {}
            attr_dict['label'] = name_single_node(node)

            if isinstance(node, Variable):
                attr_dict['shape'] = 'ellipse'
            else:
                attr_dict['shape'] = 'rectangle'
            return attr_dict

        # Traverse the graph hierarchy and plot each subgraph to a cluster
        # TODO: How do we actually create the tree structure? 
        next_tree_nodes = [(None, self.get_root_graph(), build_unique_node_name(None, None))]
        while len(next_tree_nodes) > 0:
            popped = next_tree_nodes.pop(0)
            g = popped[1]
            parent_op = popped[0]
            parent_op_name = popped[2]

            if parent_op is None:
                cluster_name = 'root'
            else:
                cluster_name = name_single_node(parent_op)
            cluster = pydot.Cluster(str(parent_op), label=cluster_name)
            
            for node in g.node_table:
                node_name = build_unique_node_name(node, parent_op)
                dot_node = pydot.Node(node_name, **name_node(node))
                cluster.add_node(dot_node)

                # Add upstream edges to the dot graph
                node_index = g.node_table[node]
                for pred in g.rxgraph.predecessors(node_index):
                    pred_name = build_unique_node_name(pred, parent_op)
                    edge = pydot.Edge(pred_name, node_name)
                    dot.add_edge(edge)

                # if g.in_degree(node) == 0:
                #     edge = pydot.Edge(parent_op_name, node_name)
                #     dot.add_edge(edge)

            for child in insights['graph_tree'][g]:
                if isinstance(child[1], Graph):
                    parent_op_name = build_unique_node_name(child[0], parent_op)
                    next_tree_nodes.append((*child, parent_op_name))

            dot.add_subgraph(cluster)
        dot.write_svg(f'{filename}.svg')
        
    def visualize_adjacency_matrix(self):
        """
        Visualizes the adjacency matrix of the graph.
        """
        self.active_graph.visualize_n2()
        
    def save_graph(self, filename: str = 'graph'):
        """Saves the graph to file

        Parameters
        ----------
        filename : str, optional
            filename to save to, by default 'graph'
        """
        self.active_graph.save(filename)


    def get_root_graph(self):
        """
        Gets the root graph.

        Returns:
            The root graph.
        """
        #TODO: TEST TEST TEST TEST

        return self.graph_tree.value

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
