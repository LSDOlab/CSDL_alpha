from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
from csdl_alpha.src.operations.derivatives.bookkeeping import listify_and_verify_variables, build_derivative_node_order, VarTangents
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from typing import Union
from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.utils.inputs import validate_and_variablize, variablize


class VJPOperation(SubgraphOperation):

    def __init__(self, graph, inputs, outputs):
        super().__init__(*inputs)
        self.name = 'vjp'

        self.assign_subgraph(graph)
        self.set_outputs(outputs)

    def compute_inline(self, *args):
        self.get_subgraph().execute_inline()

def preprocess_reverse(
        of_vars:list[Variable],
        wrt_vars:list[Variable],
        graph:Graph,
    )->dict[Variable]:
    """Computes the reverse mode derivative order of the graph and allows operations to perform any precomputations.
    """

    # Call preprocess for all variables
    # Extract subgraph of relevant nodes in the graph
    node_order = build_derivative_node_order(graph, of_vars, wrt_vars)

    # call preprocess for all operations
    for node in node_order:
        if isinstance(node, Operation):
            node.prep_vjp()
    return node_order

def vjp(seeds:list[tuple[Variable, Variable]],
        wrts:Union[Variable, list[Variable]],
        graph:Graph,
    )->dict[Variable]:
    # Preprocess inputs
    of_vars = []
    for of_var, of_seeds in seeds:
        of_vars.append(validate_and_variablize(of_var))

        # Seeds must match shape of the variable
        if of_seeds.shape != of_var.shape:
            raise ValueError(f"Seed shape {of_seeds.shape} and variable shape {of_var.shape} do not match.")

    wrt_vars = listify_and_verify_variables(wrts)
    node_order = preprocess_reverse(of_vars, wrt_vars, graph)
    return _vjp(seeds, wrt_vars, node_order)

def _vjp(seeds:list[tuple[Variable, Variable]],
        wrt_vars:Union[Variable, list[Variable]],
        node_order:list[Union[Variable,Operation]],
    )->dict[Variable]:
    """ Computes the vector-Jacobian product of the seeds with respect to the wrts in the graph.

    Parameters
    ----------
    seeds : list[tuple[Variable, Variable]]
        A list of variable and seed pairs
    wrts : Union[Variable, list[Variable]]
        A list of variables to propagate derivatives through
    graph : Graph
        The graph in which to compute the derivatives

    Returns
    -------
    dict[Variable]
        The accumulated cotangents for the wrt variables given the output seeds

    Raises
    ------
    ValueError
        Seeds must match the shape of the associated variable
    """

    # initialize seeds and final wrt cotangents
    import numpy as np
    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()

    cotangents:VarTangents = VarTangents()
    for of_var, seed in seeds:
        cotangents.initialize(of_var)
        cotangents.accumulate(of_var, variablize(seed))

        # store it for debugging purposes:
        cotangents.add_of(of_var)
    
    for wrt_var in wrt_vars:
        # store it for debugging purposes:
        cotangents.add_wrt(wrt_var)

    # perform the vector-jacobian here in terms of CSDL operations by going through the node order
    for node in node_order:
        if isinstance(node, Variable):
            cotangents.initialize(node)

    for node in node_order:
        # rec = csdl.get_current_recorder()
        # rec.visualize_graph(filename = graph.name)
        if isinstance(node, Operation):
            try:
                # Main derivative accumulation functiion.
                node.evaluate_vjp(cotangents, *node.inputs, *node.outputs)
            except Exception as e:
                if recorder.debug is True:
                    node.print_trace()
                raise ValueError(f"Error with VJP of operation {node.info()}: {e}")

            if recorder.debug is True and recorder.inline is True:
                for input in node.inputs:
                    if cotangents.check(input):
                        if cotangents[input] is not None:
                            if np.isnan(cotangents[input].value).any():
                                print(f'\n========Nan found during derivative of operation {node.info()}:=======')
                                print(f'Input variable: ({input.info()})')
                                # input.print_trace(tab = True)
                                # print('Operation trace:')
                                # node.print_trace(tab = True)
                                print()


    wrt_cotangents:dict[Variable:Variable] = {}
    for wrt_var in wrt_vars:
        wrt_cotangent = cotangents[wrt_var]
        if wrt_cotangent is None:
            wrt_cotangents[wrt_var] = None
            continue
            # wrt_cotangent = csdl.Variable(name = f'seeds_{wrt_var.name}', value = np.zeros(wrt_var.shape))
        wrt_cotangents[wrt_var] = wrt_cotangent
    return wrt_cotangents

