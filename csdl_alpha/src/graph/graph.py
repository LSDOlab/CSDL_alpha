import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import numpy as np
import csdl_alpha.utils.error_utils as error_utils
# from csdl_alpha.utils.error_utils import GraphError

class Graph():

    def __init__(self, name = None):
        self.rxgraph = rx.PyDiGraph()
        self.node_table = {}
        self.add_missing_variables = False
        if name is None:
            self.name = 'graph'
        else:
            self.name = name

    def add_node(self, node):
        if node in self.node_table:
            return
        index = self.rxgraph.add_node(node)
        self.node_table[node] = index

    def in_degree(self, node):
        return self.rxgraph.in_degree(self.node_table[node])

    def add_edge(self, node_from, node_to):
        from_ind = self.node_table[node_from]
        to_ind = self.node_table[node_to]
        self.rxgraph.add_edge(from_ind, to_ind, (from_ind, to_ind))

    def add_variable(self, variable):
        self.add_node(variable)

    def add_operation(self, operation):
        self.add_node(operation)

    def execute_inline(self, subset = None):
        """
        executes the graph inline
        """
        if subset is not None:
            if not isinstance(subset, set):
                raise TypeError("subset is not a set")

        self.check_self()
        # print('executing inline')
        sorted_nodes = rx.topological_sort(self.rxgraph)
        for node_index in sorted_nodes:
            node = self.rxgraph[node_index]

            if subset is not None:
                if self.node_table[node] not in subset:
                    continue

            # print(get_node_info_string(node, self))
            if is_operation(node):
                # print(f"Executing {node}")
                # print('\t', *[input.value for input in node.inputs])
                # print('\t', *[output.value for output in node.outputs])
                node.set_inline_values()
                # print('\t', *[output.value for output in node.outputs])

    def update_downstream(self, node):
        descendants = rx.descendants(self.rxgraph, self.node_table[node])
        self.execute_inline(subset = descendants)

    def print_all_values(self):
        """
        prints all values of variables in the graph
        """
        for node in self.rxgraph.nodes():
            if is_variable(node):
                print(f"{node.name}: {node.value}")

    def extract_subgraph(
            self,
            sources,
            targets,
            keep_variables = True,    
        ):
        """
        TODO: move this to a function?
        TODO: Have a graph_transformations.extract?
        TODO: Redo this method. it sucks

        Destroy subgraph between sources and targets and creates another graph object of that subgraph
        if keep_variables is True, then variables are not destroyed

        Returns:
        - subgraph: Graph object containing new subgraph
        - subgraph_in: Input variables to the subgraph
        - subgraph_out: Output variables to the subgraph 
        """

        # TODO: _get_intersection should return outputs of operations not in subgraph?
        #      -> like the S = S.union(subgraph_outputs) should be in _get_intersection?
        S = self._get_intersection(
            sources = sources,
            targets = targets,
        )

        # Compute the outputs to the subgraph
        subgraph_outputs = set()
        subgraph_variables = set()
        temp_indices = set()
        for node_index in S:
            node = self.rxgraph[node_index]
            if is_operation(node):
                for output in self.rxgraph.successors(node_index):
                    subgraph_outputs.add(output)
            else:
                subgraph_variables.add(node)

        subgraph_inputs = subgraph_variables.symmetric_difference(subgraph_outputs)

        # Create a new graph which is the subgraph
        rx_sg = self.rxgraph.subgraph(list(S))

        subgraph = Graph()
        subgraph.rxgraph = rx_sg
        subgraph.update_node_table()


        # Delete the nodes in the subgraph from the graph
        delete_nodes = set()
        for node_index in S:
            node = self.rxgraph[node_index]
            if keep_variables:
                if is_variable(node):
                    continue
            delete_nodes.add(node_index)
    

        # self.visualize(f'in_extract_b')

        # Checks: (temporary)
        # TODO: apply checks in debug mode?
        if 0:
            print("\nChecking extraction...")
            # all inputs and outputs should be in self
            subgraph_inputs_and_outputs = subgraph_inputs.union(subgraph_outputs)
            for node in subgraph_inputs_and_outputs:
                if node not in self.rxgraph.nodes():
                    raise ValueError(f"Node {node} not in graph {get_node_info_string(node, self)}")
            print("  - Check 1 passed")

            # all inputs and outputs should be in subgraph and no other variables
            for node in subgraph_inputs_and_outputs:
                if node not in subgraph.rxgraph.nodes():
                    raise ValueError(f"Node {node} not in subgraph {get_node_info_string(node, self)}")
            # too lazy to check for no other variables part
            print("  - Check 2 passed")

            # all inputs in subgraph should have no predecessors
            for node in subgraph_inputs:
                if len(subgraph.rxgraph.predecessors(subgraph.node_table[node])) != 0:
                    raise ValueError(f"Input {node} has predecessors {get_node_info_string(node, self)}")
            print("  - Check 3 passed")

            print("Check complete...\n")

        self._delete_nodes(delete_nodes)
        # self.visualize(f'in_extract_a')

        return subgraph, subgraph_inputs, subgraph_outputs

    # TODO: make this work with variables only?
    def _delete_nodes(self, nodes):
        """
        Deletes nodes from the graph
        """
        from csdl_alpha.src.graph.node import Node
        for node_index in nodes:
            if isinstance(node_index, Node):
                node_index = self.node_table[node_index]
            self.rxgraph.remove_node(node_index)
        self.update_node_table()

    def _replace_node(self, old_node, new_node):
        """
        Replaces old_node with new_node in the graph
        """
        # replace node in graph
        self._delete_nodes([new_node]) #NOTE: this is kinda dumb
        old_node_index = self.node_table[old_node]
        self.rxgraph[old_node_index] = new_node
        self.update_node_table()
        # TODO: update operations to refer to new node?
        for operation in self.rxgraph.successors(old_node_index):
            print(operation)
            for i in range(len(operation.inputs)):
                if operation.inputs[i] == old_node:
                    operation.inputs[i] = new_node
        
    def check_self(self):
        """
        raise error if the graph node table and nodes are not synced correctly
        """
        
        # nodes in node_table should be in graph
        nodes = set(self.rxgraph.nodes())
        for node, index in self.node_table.items():
            if node not in nodes:
                raise ValueError(f"Node {node} not in graph")
            if self.rxgraph[index] != node:
                raise ValueError(f"Node {node} not in graph")
            
        # nodes in graph should be in node_table
        if len(nodes) != len(self.node_table):
            raise ValueError(f"Node table (length {len(self.node_table)}) and graph nodes (length {len(nodes)}) are not synced")
        
        # If synced, we're good here
        # print("Graph and node table are synced!!!Graph and node table are synced!!!Graph and node table are synced!!!Graph and node table are synced!!!")

    def update_node_table(self):
        """
        Update the node table with the rustworkx graph
        """
        self.node_table = {}
        for index in self.rxgraph.node_indices():
            node = self.rxgraph[index]
            self.node_table[node] = index

        self.check_self()

    def _get_intersection(
            self,
            sources,
            targets,
            check_sources = True,
            check_targets = True,
        ):
        """
        Returns all nodes between sources and targets.
        If check_sources is True, then checks to make sure all sources should affect atleast one target
        If check_targets is True, then checks to make sure all targets should be affected by atleast one source
        """

        # D = Union of all source descendants
        # A = Union of all target ancestors
        # S = D intersection A

        # Get the sources and targets indices
        source_indices = [self.node_table[source] for source in sources]
        target_indices = [self.node_table[target] for target in targets]

        # Get the nodes between the sources and targets
        D = set()
        for source_index in source_indices:
            D = D.union(rx.descendants(self.rxgraph, source_index))

        A = set()
        for target_index in target_indices:
            A = A.union(rx.ancestors(self.rxgraph, target_index))
            A.add(target_index)

        if check_sources:
            for source_index in source_indices:
                if source_index not in A:
                    targets_string = error_utils.get_node_name_string(targets)
                    raise error_utils.GraphError(
                        f"Source node {self.rxgraph[source_index].name} does not affect any target node(s) {targets_string}",
                        tag = 'no_path',
                        relevant_nodes = self.rxgraph[source_index],
                    )
        if check_targets:
            for target_index in target_indices:
                if target_index not in D:
                    sources_string = error_utils.get_node_name_string(sources)
                    raise error_utils.GraphError(
                        f"Target node {self.rxgraph[target_index].name} is not affected by any source node(s) {sources_string}",
                        tag = 'no_path',
                        relevant_nodes = self.rxgraph[target_index],
                    )


        # Find intersection
        S = D.intersection(A)

        # S contains operation nodes which could be leaf nodes in the graph
        # Therefore, add all inputs/outputs to the operations to the graph
        pred_succ_vars = set()
        for node_index in S:
            node = self.rxgraph[node_index]
            if is_operation(node):
                for input_node in self.rxgraph.predecessors(self.node_table[node]):
                    pred_succ_vars.add(self.node_table[input_node])
                for output_node in self.rxgraph.successors(self.node_table[node]):
                    pred_succ_vars.add(self.node_table[output_node])
        S = S.union(pred_succ_vars)
        return S

    def visualize(self, filename = 'image'):
        from csdl_alpha.src.graph.variable import Variable
        inverse_node_table = {v: k for k, v in self.node_table.items()}

        # def name_node(node_index):
        #     print(node_index)
        #     attr_dict = {}
        #     node = inverse_node_table[node_index]
        #     if isinstance(node, Variable):
        #         attr_dict['label'] = node.name
        #     else:
        #         attr_dict['label'] = node.name
        #     return attr_dict


        dot = self.to_dot(node_attr_fn=self.name_node)
        dot.write_svg(f'{filename}.svg')


        # graphviz_draw(self, node_attr_fn = self.name_node, filename= 'graph.png')

    def replace(self, graph):
        """
        Replace the current graph with the graph passed in
        """
        self.rxgraph = graph.rxgraph
        self.update_node_table()

    def to_dot(self, node_attr_fn=None):
        import pydot


        dot = pydot.Dot(graph_type='digraph')

        # Create a dictionary to store subgraphs
        subgraphs = {}

        # For each node in the graph
        for node in self.rxgraph.nodes():
            # Get the namespace of the node
            namespace_obj = node.namespace if hasattr(node, 'namespace') else 'global'
            namespace = namespace_obj.name
            if namespace is None:
                namespace = 'global'

            # If the namespace is not in the subgraphs dictionary, create a new subgraph for it
            if namespace_obj not in subgraphs:
                subgraphs[namespace_obj] = pydot.Cluster(namespace, label=namespace)
                if namespace_obj.parent is not None:
                    if namespace_obj.parent not in subgraphs:
                        if namespace_obj.parent.name is None:
                            parent_name = 'global'
                        else:
                            parent_name = namespace_obj.parent.name
                        subgraphs[namespace_obj.parent] = pydot.Cluster(parent_name, label=parent_name)
                    subgraphs[namespace_obj.parent].add_subgraph(subgraphs[namespace_obj])

            # Create a new node for the dot graph
            dot_node = pydot.Node(str(node), **node_attr_fn(node))

            # Add the node to the corresponding subgraph
            subgraphs[namespace_obj].add_node(dot_node)

        # Add all edges to the dot graph
        for edge_tuple in self.rxgraph.edge_index_map().values():
            dot.add_edge(pydot.Edge(str(self.rxgraph[edge_tuple[0]]), str(self.rxgraph[edge_tuple[1]])))

        # add subgraphs to subgraphs to reflect namespace tree
        


        # Add all subgraphs to the dot graph
        for namespace, subgraph in subgraphs.items():
            if namespace.parent is None:
                dot.add_subgraph(subgraph)

        return dot

    def name_node(self, node):
        attr_dict = {}
        if node.name is None:
            attr_dict['label'] = 'var'
        else:
            attr_dict['label'] = node.name
        return attr_dict

    def create_n2(self):
        from csdl_alpha.src.graph.variable import Variable
        # Get the number of nodes in the graph
        n = len(self.rxgraph.nodes())

        # Create an empty N2 matrix
        n2_matrix = np.zeros((n, n))

        # For each node in the graph
        for node in range(n):
            # Check if the node is a variable node
            if isinstance(self.rxgraph[node], Variable):
                n2_matrix[node, node] = 0.5
                # Get the successors of the node
                successor_ops = self.rxgraph.successor_indices(node)

                # For each successor of the node
                for successor_op in successor_ops:
                    for successor in self.rxgraph.successor_indices(successor_op):
                        # Mark the corresponding cell in the N2 matrix
                        n2_matrix[node, successor] = 1
                        # Remove all zero rows and columns from n2_matrix

        n2_matrix = n2_matrix[~np.all(n2_matrix == 0, axis=1)]
        n2_matrix = n2_matrix[:, ~np.all(n2_matrix == 0, axis=0)]
        # Return the N2 matrix
        return n2_matrix

    def visualize_n2(self):
        import matplotlib.pyplot as plt
        from csdl_alpha.src.graph.variable import Variable
        # Create the N2 matrix
        n2_matrix = self.create_n2()
        node_names = []
        for node in self.rxgraph.nodes():
            if isinstance(node, Variable):
                node_names.append(self.name_node(node)['label'])

        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Display an image on the axes
        cax = ax.matshow(n2_matrix, cmap='gray_r')

        # Set the labels for the x and y axes
        ax.set_xticks(np.arange(len(node_names)))
        ax.set_yticks(np.arange(len(node_names)))
        ax.set_xticklabels(node_names)
        ax.set_yticklabels(node_names)

        # Rotate the x labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Show the plot
        plt.show()

def is_operation(x):
    from csdl_alpha.src.graph.operation import Operation
    return isinstance(x, Operation)

def is_variable(x):
    from csdl_alpha.src.graph.variable import Variable
    return isinstance(x, Variable)


def get_node_info_string(node, graph):

    # graph.visualize()
    node_info = "\n\n"
    if is_variable(node):
        node_type = 'Variable'
    elif is_operation(node):
        node_type = 'Operation'
    
    node_preds = ""
    for pred in graph.rxgraph.predecessors(graph.node_table[node]):
        node_preds += f"{pred.name}, "
    
    node_succs = ""
    for succ in graph.rxgraph.successors(graph.node_table[node]):
        node_succs += f"{succ.name}, "

    node_info += f"\nName:                  {node.name}"
    node_info += f"\nNode:                  {node}"
    node_info += f"\nType:                  {node_type}"
    node_info += f"\nPredecessors:          {node_preds}"
    node_info += f"\nSuccessors:            {node_succs}"
    if is_variable(node):
        node_info += f"\nValue:                 {node.value}"
    return node_info