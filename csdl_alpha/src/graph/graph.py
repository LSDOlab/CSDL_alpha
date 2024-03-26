import rustworkx as rx
from rustworkx.visualization import graphviz_draw
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

        def name_node(node):
            attr_dict = {}
            if node.name is None:
                attr_dict['label'] = 'test'
            else:
                attr_dict['label'] = node.name
            return attr_dict


        graphviz_draw(self, node_attr_fn = name_node, filename= 'graph')