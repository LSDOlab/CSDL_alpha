import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import numpy as np

class Graph(rx.PyDiGraph):
    def __init__(self):
        super().__init__()
        self.node_table = {}
    
    def add_node(self, node):
        if node in self.node_table:
            return
        index = super().add_node(node)
        self.node_table[node] = index

    def add_edge(self, node_from, node_to):
        from_ind = self.node_table[node_from]
        to_ind = self.node_table[node_to]
        super().add_edge(from_ind, to_ind, (from_ind, to_ind))

    def add_variable(self, variable):
        self.add_node(variable)

    def add_operation(self, operation):
        self.add_node(operation)

    def visualize(self):
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
        dot.write_png('image.png')


        # graphviz_draw(self, node_attr_fn = self.name_node, filename= 'graph.png')

    def to_dot(self, node_attr_fn=None):
        import pydot


        dot = pydot.Dot(graph_type='digraph')

        # Create a dictionary to store subgraphs
        subgraphs = {}

        # For each node in the graph
        for node in self.nodes():
            # Get the namespace of the node
            namespace_obj = node.namespace if hasattr(node, 'namespace') else 'global'
            namespace = namespace_obj.name
            if namespace is None:
                namespace = 'global'

            # If the namespace is not in the subgraphs dictionary, create a new subgraph for it
            if namespace_obj not in subgraphs:
                subgraphs[namespace_obj] = pydot.Cluster(namespace, label=namespace)
                if namespace_obj.parent is not None:
                    subgraphs[namespace_obj.parent].add_subgraph(subgraphs[namespace_obj])

            # Create a new node for the dot graph
            dot_node = pydot.Node(str(node), **node_attr_fn(node))

            # Add the node to the corresponding subgraph
            subgraphs[namespace_obj].add_node(dot_node)

        # Add all edges to the dot graph
        for edge in self.edges():
            dot.add_edge(pydot.Edge(str(self[edge[0]]), str(self[edge[1]])))

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
        n = len(self.nodes())

        # Create an empty N2 matrix
        n2_matrix = np.zeros((n, n))

        # For each node in the graph
        for node in range(n):
            # Check if the node is a variable node
            if isinstance(self[node], Variable):
                n2_matrix[node, node] = 0.5
                # Get the successors of the node
                successor_ops = self.successor_indices(node)

                # For each successor of the node
                for successor_op in successor_ops:
                    for successor in self.successor_indices(successor_op):
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
        for node in self.nodes():
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