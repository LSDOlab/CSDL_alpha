from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node

def get_subgraph():
    return

def extract():
    return

def loopify_subgraph(
        stacked_inputs:dict[Variable, Variable],
        outputs:dict,
    ):
    """
    Takes a part of a graph that exists in a graph and turns it into a loop:

       (loop)
    o-->o-->o-->o
    1   2   3   4
    
    becomes

    1       4
    o-->L-->o
        |
     (o-->o)
      2   3
     
    NOTE: It's difficult to deal with outputs: Would the outputs be the last iteration version? the average?

    Args:
        stacked_inputs: dict[Variable, Variable]
            Map of input variables in subgraph to stacked inputs with extra first dimension 
        
        outputs: 
            Mapping outputs to inputs  
    """

    # Pre-processing:
    # Make sure inputs and outputs are in the graph
    # subgraph must be convex (dont know better terminology???) in the outer graph. This prevents loops in the DAG
    # subgraph is built from the stacked inputs and outputs

    # 1) extract the subgraph:
    # - Make a subgraph from the outer graph
    # - - Make sure needed extra variables are inserted
    # - Delete the subgraph from the outer graph
    # - - Make sure to not delete input/outputs from the outer graph

    # 2) Create and insert loop operations of the subgraph
    # - The loop has to index the stacked inputs
    # - Insert the loop operation back into the outer graph
    # - for provided stacked inputs, we need to delete 

    # 3) Connect the outputs to the outer graph

    # Pre-processing
    import csdl_alpha as csdl
    rec = csdl.get_current_recorder()
    current_graph = rec.active_graph

    sources = []
    targets = []
    num_iter = None

    for input_var, stacked_input in stacked_inputs.items():

        input_var:Variable
        stacked_input:Variable

        # Check to make sure input_var is in the graph
        if input_var not in current_graph.node_table:
            raise ValueError(f'Input variable {input_var.info()} not in the graph')
      
        # Check to make sure stacked_input is in the graph
        if stacked_input not in current_graph.node_table:
            raise ValueError(f'Stacked input variable {stacked_input.info()} not in the graph')
        
        # Check to make sure stacked_inputs is a valid stack of input_var
        if num_iter is None:
            num_iter = stacked_input.shape[0]
        elif num_iter != stacked_input.shape[0]:
            raise ValueError(f'Stack size for stacked_input {input_var.info()} does not match previous size. {num_iter} != {stacked_input.shape[0]}')

        # Check to make sure the other shapes match
        if input_var.shape != stacked_input.shape[1:]:
            raise ValueError(f'Shape mismatch between input_var {input_var.info()} and stacked_input {stacked_input.info()}. {input_var.shape} != {stacked_input.shape[1:]}')

        # record
        sources.append(input_var)

    for output_var in outputs:
        if output_var not in current_graph.node_table:
            raise ValueError(f'Output variable {output_var} not in the graph')
        else:
            print(f'Found output variable {output_var} in the graph')
        targets.append(output_var)

    rec.visualize_graph('1', visualize_style='hierarchical')
    subgraph, subgraph_inputs, subgraph_outputs = current_graph.extract_subgraph(
        sources,
        targets,
        keep_variables=True
    )
    rec.visualize_graph('2', visualize_style='hierarchical')

    subgraph.visualize()

    # with csdl.experimental.enter_loop() as vjp_loop_builder:
    #     pass

    exit()
    return