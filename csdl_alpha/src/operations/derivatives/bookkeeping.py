from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from typing import Union
from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.utils.inputs import validate_and_variablize

def build_derivative_node_order(
        graph:Graph,
        ofs: list[Variable],
        wrts: list[Variable],
        reverse: bool = True,
    )->list[Variable]:
    """
    Gets the subgraph of the graph that is relevant for the derivative of the output variables with respect to the input variables.
    """
    
    import rustworkx as rx
    
    # Find the subgraph of ofs, wrts
    intersecting_nodes = graph._get_intersection(
        wrts,
        ofs,
        check_sources=False,
        check_targets=False,
        add_hanging_input_variables=False,
        add_hanging_output_variables=False,
    )
    for of_var in ofs:
        intersecting_nodes.add(graph.node_table[of_var])
    for wrt_var in wrts:
        intersecting_nodes.add(graph.node_table[wrt_var])
    # descendants = set()
    # wrt_var_indices = set()
    # for wrt_var in wrts:
    #     descendants.update(rx.descendants(graph.rxgraph, graph.node_table[wrt_var]))
    #     wrt_var_indices.add(graph.node_table[wrt_var])
    # relevant_nodes = rx.ancestors(graph.rxgraph, graph.node_table[of_var]).intersection(descendants)
    # relevant_nodes.add(graph.node_table[of_var])
    # relevant_nodes.update(wrt_var_indices)

    # Compute the order of nodes to process
    node_order = []
    if reverse:
        ordered = reversed(rx.topological_sort(graph.rxgraph))
    else:
        ordered = rx.topological_sort(graph.rxgraph)

    for node in ordered:
        node_index = node
        if node_index in intersecting_nodes:
            node_order.append(graph.rxgraph[node])
    return node_order

def listify_and_verify_variables(variables:Union[Variable, list[Variable]])->list[Variable]:
    if isinstance(variables, (list, tuple)):
        variables = [validate_and_variablize(var) for var in variables]
    else:
        variables = [validate_and_variablize(variables)]
    return variables


class VarTangents():
    def __init__(self):
        """Store cotangents/tangents for variables in the graph.
        """
        self.tangent_dictionary:dict[Variable: Variable] = {}

        self.ofs = []
        self.wrts = []

    def add_of(self, of:Variable):
        """Doesn't do anything practical, just for fun.
        """
        self.ofs.append(of)

    def add_wrt(self, wrt:Variable):
        """Doesn't do anything practical, just for fun.
        """
        self.wrts.append(wrt)

    def accumulate(self, variable:Variable, tangent:Variable)->None:
        """Accumulate a tangent for a variable.

        Args:
            variable (Variable): The variable to accumulate the tangent for.
            tangent (Variable): The tangent to accumulate.
        """
        if tangent is None:
            print(f"Warning: Tangent for {variable} is None.")
            for wrt in self.wrts:
                print(wrt.info())

        if variable.shape != tangent.shape:
            raise ValueError(f"Variable shape {variable.shape} and (co)tangent shape {tangent.shape} do not match.")
        if variable in self.tangent_dictionary:
            if self.tangent_dictionary[variable] is None:
                self.tangent_dictionary[variable] = tangent
            elif isinstance(tangent, Variable):
                self.tangent_dictionary[variable] = self.tangent_dictionary[variable] + tangent
            else:
                raise TypeError(f"Expected tangent to be a Variable, but got {type(tangent)}.")
        else:
            raise KeyError(f"Variable {variable} not found in tangent dictionary.")

        self.tangent_dictionary[variable].add_name(f'tangent_{variable.name}')

    def initialize(self, variable:Variable):
        """Initialize the tangent for a variable.

        Args:
            variable (Variable): The variable to initialize the tangent for.
        """
        if variable not in self.tangent_dictionary:
            import numpy as np
            self.tangent_dictionary[variable] = None
            # self.tangent_dictionary[variable].add_name(f'tangent_{variable.name}')

    def check(self, variable:Variable):
        """Check if a variable has a tangent.

        Args:
            variable (Variable): The variable to check.

        Returns:
            bool: True if the variable has a tangent.
        """
        return variable in self.tangent_dictionary

    def __getitem__(self, variable:Variable):
        """Get the tangent for a variable.

        Args:
            variable (Variable): The variable to get the tangent for.

        Returns:
            Variable: The tangent for the variable.
        """
        return self.tangent_dictionary[variable]
