import rustworkx as rx

class Graph(rx.PyDiGraph):
    def __init__(self):
        super().__init__()
        self.node_table = {}
    
    def add_node(self, node):
        index = super().add_node(node)
        self.node_table[node] = index

    def add_variable(self, variable):
        self.add_node(variable)

    def add_operation(self, operation):
        self.add_node(operation)