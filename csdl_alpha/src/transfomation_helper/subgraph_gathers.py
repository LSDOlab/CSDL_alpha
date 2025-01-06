from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node

def get_subgraph():
    return

def extract():
    return

def loopify_subgraph(
        subgraph,
        stacked_inputs:list[Variable],
        outputs:dict,
    ):
    """
    NOTE: It's difficult to deal with outputs: Would the outputs be the last iteration version? the average?

    Args:
        subgraph:

        stacked_inputs: dict[Variable, Variable]
            Map of input variables in subgraph to stacked inputs with extra first dimension 
        
        outputs: 
            Map of output variables to their original outputs
    """

    

    return