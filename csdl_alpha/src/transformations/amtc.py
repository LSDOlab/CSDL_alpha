from csdl_alpha.src.transformations.transformation import TransformationBase
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import get_type_string, variablize

import networkx as nx
import rustworkx as rx
import numpy as np
import string

from typing import Any

alphabet = string.ascii_lowercase

class AMTC(TransformationBase):

    def apply(self, rvs:dict[Variable, VariableLike], outputs:list[Variable], debug:bool=False)->dict[Variable, Variable]:
        recorder = self.get_current_recorder()
        graph = recorder.get_root_graph()

        import csdl_alpha as csdl
        ct = csdl.transformation_helper

        # Checks:
        # 1) Make sure all rvs, stacked inputs, outputs are in the graph
        # 2) Make sure quadrature points are in the correct shape as outputs
        for rv, qp in rvs.items():
            if not ct.is_var(rv):
                raise TypeError(f'Random variables must be of type Variable, {get_type_string(rv)} given')
            if not rv in graph.node_table:
                raise KeyError(f'Random variable {rv.info()} not in recorder root graph')
            
            qp = variablize(qp)
            if qp.shape[1:] != rv.shape:
                expected_shape = f"(n, {', '.join(map(str, rv.shape))})"
                raise ValueError(f'Quadrature points of random variable {rv.info()} expected to have shape {expected_shape}, but has shape {qp.shape}')
            rvs[rv] = qp

        # Steps:
        # 1) Create a partition graph
        # 2) For each partition:
        #    - loopify subgraph
        #    - create einsums

        # Create partition:
        # 1a) first create a list of all possible partitions
        # 1b) For each RV, figure out all depend nodes
        # 1c) Loop through each node:
        # - - Assign node to partition
        # - - If predecessors are in different partitions, save as input to partition and add th predecessors as outputs to their partitions

        # 1a) Create a list of all possible partitions
        source_indices_to_rv:dict[int,Variable] = dict()
        rvs_info:dict[Variable,int] = dict() # Maps random variables to source integer and more info
        for i, rv in enumerate(rvs):
            # CHECKS:
            # Check to make sure rv is not dependent on any other rv
            if graph.in_degree(rv) > 0:
                raise ValueError(f'RV {rv.info()} is a dependent variable')

            # Organization
            source_indices_to_rv[i] = rv # Maps integers to random variables
            rvs_info[rv] = {
                'source_ind':i,
                'qp':rvs[rv],
            }

        # Initialize partition graph
        partition_graph = build_partition_graph(
            source_indices_to_rv,
            rvs_info,
        )

        # For each random variable, figure out dependencies
        output_descendants = set()
        output_set = set(outputs)
        for output in outputs:
            output_descendants_ind = rx.ancestors(graph.rxgraph, graph.node_table[output])
            output_descendants = output_descendants.union({graph.rxgraph.nodes()[ind] for ind in output_descendants_ind})
            # output_descendants = ({graph.rxgraph.nodes()[ind] for ind in output_descendants_ind})
            output_descendants.add(output)
        dependency_data:dict[Variable, set[int]] = {node:set() for node in output_descendants}
        dependency_data.update({rv:set() for rv in rvs_info})
        for index in source_indices_to_rv:
            assign_dependencies(
                index,
                recorder,
                dependency_data,
                source_indices_to_rv,
            )
        for node, index_set in dependency_data.items():
            dependency_data[node] = frozenset(index_set)

        # figure out inputs and outputs for each partition.
        # We do this by looping through each node and checking if the predecessors are in the partition.
        # If the predecessors are NOT in the partition,
        # then the predecessor must be an output of the previous partition and expanded as an input to the current partition.
        index_sets_to_nodes = {frozenset(index_tuple):index_tuple for index_tuple in partition_graph.nodes()}
        for current_node in dependency_data:
            # print('current_node:', current_node, dependency_data[current_node])
            current_source_dependencies = dependency_data[current_node]
            current_partition = index_sets_to_nodes[current_source_dependencies]
            partition_graph.nodes[current_partition]['intermediates'].add(current_node)
            if isinstance(current_node, Variable):
                if graph.in_degree(current_node) == 0:
                    input_partition = index_sets_to_nodes[dependency_data[current_node]]
                    partition_graph.nodes[input_partition]['inputs'].add(current_node)
                    partition_graph.nodes[input_partition]['intermediates'].add(current_node)
                if graph.out_degree(current_node) == 0:
                    output_partition = index_sets_to_nodes[dependency_data[current_node]]
                    partition_graph.nodes[output_partition]['outputs'].add(current_node)
                    partition_graph.nodes[output_partition]['intermediates'].add(current_node)

                if current_node in output_set:
                    partition_graph.nodes[current_partition]['outputs'].add(current_node)
                continue

            for parent_variable in graph.predecessors(current_node):
                predecessor_source_dependencies = dependency_data[parent_variable]
                if current_source_dependencies != predecessor_source_dependencies:
                    pred_partition = index_sets_to_nodes[predecessor_source_dependencies]
                    partition_graph.edges[pred_partition, current_partition]['transfers'][parent_variable] = None
                    partition_graph.nodes[pred_partition]['outputs'].add(parent_variable)
                    partition_graph.nodes[pred_partition]['intermediates'].add(parent_variable)

                    partition_graph.nodes[current_partition]['inputs'].add(parent_variable)
                    partition_graph.nodes[current_partition]['intermediates'].add(parent_variable)

        if debug:
            # Lets visualize the whole graph where nodes are highlighted by partition
            containers = {}
            colors = {}
            for partition in nx.topological_sort(partition_graph):
                for node in partition_graph.nodes[partition]['inputs']:
                    if node in colors and colors[node] == 'coral': colors[node] = 'darkolivegreen1'
                    else: colors[node] = 'aquamarine'

                for node in partition_graph.nodes[partition]['outputs']:
                    if node in colors and colors[node] == 'aquamarine': colors[node] = 'darkolivegreen1'
                    else: colors[node] = 'coral'
            
                for node in partition_graph.nodes[partition]['intermediates']:
                    if node not in containers: containers[node] = [str(partition)]
                    else: containers[node].append(str(partition))

            recorder.get_root_graph().visualize(
                filename='original_graph_partitioned',
                containers=containers,
                colors=colors,
            )

            # exit('debug')


        tensor_grid_evaluations = {}
        # Loop through each partition and loopify the subgraph
        for partition in nx.topological_sort(partition_graph):
            print('partition:', partition)
            print('\tinputs:', len(partition_graph.nodes[partition]['inputs']))
            print('\toutputs:', len(partition_graph.nodes[partition]['outputs']))

            partition_outputs:set[Variable] = partition_graph.nodes[partition]['outputs']
            partition_intermediates:set[Node] = partition_graph.nodes[partition]['intermediates']

            # If there are no inputs, there is nothing in the loop --> we don't do anything
            if len(partition_graph.nodes[partition]['inputs']) == 0:
                if len(partition_outputs) > 0:
                    raise ValueError('Partition has no inputs but has outputs')
                continue
            
            # If NULL partition, no need to loopify, we keep it as is
            if len(partition) == 0:
                stacked_outputs = {}
                for output in partition_outputs:
                    action_string_pre = alphabet[1:len(output.shape)+1]
                    action_string_post = alphabet[:len(output.shape)+1]
                    action_string = f'{action_string_pre}->{action_string_post}'
                    stacked_outputs[output] = output.expand((1,)+output.shape, action = action_string)
            else:
                # Here we:
                # 1) Retrieve any expanded transfer inputs
                # 2) loopify the subgraph
                # 3) expand the transfer outputs
                partition_input_mapping = {}
                fixed_inputs = set()
                for in_edge in partition_graph.in_edges(partition):
                    pred_partition = in_edge[0]
                    for transfer_var, stacked_transfer in partition_graph.edges[pred_partition, partition]['transfers'].items():
                        if len(pred_partition) != 0:
                            partition_input_mapping[transfer_var] = stacked_transfer
                        else:
                            fixed_inputs.add(transfer_var)
                if len(partition) == 1:
                    source_input = source_indices_to_rv[list(partition)[0]]
                    partition_input_mapping[source_input] = rvs_info[source_input]['qp']
                stacked_outputs = ct.loopify_subgraph(
                    partition_input_mapping,
                    partition_outputs,
                    partition_intermediates,
                    fixed_inputs = fixed_inputs,
                    name = str(partition))

            for out_edge in partition_graph.out_edges(partition):
                succ_partition = out_edge[1]
                for transfer_var in partition_graph.edges[partition, succ_partition]['transfers']:

                    stacked_transfer_pre_expand = stacked_outputs[transfer_var]
                    if len(partition) != 0:
                        expanded_transfer = partition_graph.edges[partition, succ_partition]['expansion_function'](stacked_transfer_pre_expand)
                    else:
                        expanded_transfer = stacked_transfer_pre_expand
                    partition_graph.edges[partition, succ_partition]['transfers'][transfer_var] = expanded_transfer
                    # partition_graph.edges[partition, succ_partition]['transfers'][transfer_var] = stacked_transfer_pre_expand

            # Add variables of interest to tensor grid evaluations
            for voi in outputs:
                if voi in partition_outputs:
                    tensor_grid_evaluations[voi] = stacked_outputs[voi]
            
        # Check to make sure all outputs are in tensor_grid_evaluations
        for voi in outputs:
            if not voi in tensor_grid_evaluations:
                raise ValueError(f'INTERNAL ERROR: Output {voi.info()} not computed in any partition')
        # Now we need to delete nodes that are not in the dependency data, as this results in floating subgraphs:
        # Variables/operations that don't depend on any rvs are not added to any loops, these leaf nodes have no values
        # and cannot be evaluated
        # recorder.visualize_graph('partition_graph', visualize_style='hierarchical')
                    
        ct.delete_unvalued_descendants()
        # recorder.visualize_graph('partition_graph2', visualize_style='hierarchical')

        if debug:
            recorder.visualize_graph(
                filename='AMTCed_graph_partitioned',
                visualize_style='hierarchical',
            )

        return tensor_grid_evaluations

def build_partition_graph(
        source_indices_to_rv:dict[Variable, Variable],
        rvs_info:dict[Variable,tuple[Any]],
    ):

    partition_graph = nx.DiGraph()

    # Given integers, create a list of sets of all possible combinations of the integers
    # for example, [1,2,3] -> [{1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}]
    from itertools import chain, combinations
    s = list(source_indices_to_rv.keys())
    powerset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))    
    
    # Build nodes
    for subset in powerset:
        partition_graph.add_node(subset)
        partition_graph.nodes[subset]['inputs'] = set()
        partition_graph.nodes[subset]['outputs'] = set()
        partition_graph.nodes[subset]['intermediates'] = set()

    # Build edges (n**2 algorithm)
    for subset_pred in powerset:
        set_pred = set(subset_pred)
        for subset_succ in powerset:
            set_succ = set(subset_succ)
            if set_pred.issubset(set_succ) and set_pred != set_succ:
                partition_graph.add_edge(subset_pred, subset_succ)

                # Attributes:
                partition_graph.edges[subset_pred, subset_succ]['expansion_function'] = build_expansion_func(
                    subset_pred,
                    subset_succ,
                    source_indices_to_rv,
                    rvs_info,
                )
                partition_graph.edges[subset_pred, subset_succ]['transfers'] = {}
    
    # Visualize networkx graph with labels corresponding to node hash
    if 0:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(partition_graph)
        nx.draw(partition_graph, pos, with_labels=True)
        plt.show()

    return partition_graph

def assign_dependencies(
        index:int,
        recorder,
        dependency_data:dict[Variable, set[int]],
        source_indices_to_rv:dict[int,Variable],
    ):
    import csdl_alpha as csdl
    recorder:csdl.Recorder = recorder

    # Get descendants
    source_variable = source_indices_to_rv[index]
    rx_index = recorder.get_root_graph().node_table[source_variable]
    descendants = rx.descendants(
        recorder.get_root_graph().rxgraph, rx_index)
    for desc_ind in descendants:
        desc = recorder.get_root_graph().rxgraph.nodes()[desc_ind]
        if desc in dependency_data:
            dependency_data[desc].add(index)
    dependency_data[source_variable].add(index)

def build_expansion_func(
        subset_pred:tuple[int],
        subset_succ:tuple[int],
        source_indices_to_rv:dict[int,Variable],
        rvs_info:dict[Variable,tuple[Any]],
    ):
    import csdl_alpha as csdl

    input_num_nodes = []
    for rv_ind in subset_pred:
        rv = source_indices_to_rv[rv_ind]
        num_1d_qp = rvs_info[rv]['qp']
        input_num_nodes.append(num_1d_qp.shape[0])

    output_num_nodes = []
    for rv_ind in subset_succ:
        rv = source_indices_to_rv[rv_ind]
        num_1d_qp = rvs_info[rv]['qp']
        output_num_nodes.append(num_1d_qp.shape[0])

    output_n_1d_qp = output_num_nodes
    input_n_1d_qp = input_num_nodes
    input_dp_index = subset_pred
    result_dp_index = subset_succ

    def expansion_func(x):
        # Figure out number of quadrature points
        n_qp = 1
        for i in output_n_1d_qp:
            n_qp = n_qp*i

        index_letter_list = list(string.ascii_lowercase)
        einsum_input_string1 = ''
        einsum_input_string2 = ''
        einsum_output_string = ''
        expand_index = []
        for index in result_dp_index:
            if index in input_dp_index:
                einsum_input_string1 += index_letter_list[index]
            if index not in input_dp_index:
                einsum_input_string2 += index_letter_list[index]
                expand_index.append(index)
            einsum_output_string += index_letter_list[index]

        # einsum_input_string1 += '...'
        # einsum_input_string2 += '...'
        # einsum_output_string += '...'
        suffix_possible_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        x_orig_shape = x.shape[1:]
        x_orig_ndim = len(x_orig_shape)
        suffix_letters = suffix_possible_letters[:x_orig_ndim]
        einsum_input_string1 += suffix_letters
        einsum_input_string2 += suffix_letters
        einsum_output_string += suffix_letters

        # Out shape is first dimension expanded with correct size
        out_shape = (n_qp,)+x.shape[1:]

        matrix1_size = []
        for i in expand_index:
            index = result_dp_index.index(i)
            matrix1_size.append(output_n_1d_qp[index])
            #print('missing index', index)
        # matrix1_size.append(1)
        matrix1_size = matrix1_size + list(x_orig_shape)

        in_shape = tuple(num_qp for num_qp in input_n_1d_qp) + x_orig_shape
        
        # Prints:
        if 0:
            print('####################')
            print('dependency transfer:', len(input_dp_index), '->', len(result_dp_index))
            print(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}')
            print('shapes:', x.shape,'-(reshape)->', in_shape ,f'-(einsum w/ {tuple(matrix1_size)})->', out_shape)
            print()
        return csdl.reshape(
            csdl.einsum(
                csdl.reshape(x, in_shape),
                np.ones(tuple(matrix1_size)),
                action=f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}'
            ),
            out_shape)
        # OLD:

        index_letter_list = list(string.ascii_lowercase)
        n_qp = 1
        for i in output_n_1d_qp:
            n_qp = n_qp*i
        einsum_input_string1 = ''
        einsum_input_string2 = ''
        einsum_output_string = ''
        expand_index = []
        for index in result_dp_index:
            if index in input_dp_index:
                einsum_input_string1 += index_letter_list[index]
            if index not in input_dp_index:
                einsum_input_string2 += index_letter_list[index]
                expand_index.append(index)
            einsum_output_string += index_letter_list[index]
        einsum_input_string1 += '...'
        einsum_input_string2 += '...'
        einsum_output_string += '...'
        matrix1_size = []
        for i in expand_index:
            index = result_dp_index.index(i)
            matrix1_size.append(output_n_1d_qp[index])
            #print('missing index', index)
        matrix1_size.append(1)
        out_shape = list(x.shape)
        out_shape[0] = n_qp
        out_shape = tuple(out_shape)
        print('####################')
        print('input index', input_dp_index)
        print('output index', result_dp_index)
        print(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}')
        print('x shape:', x.shape)
        print(tuple(matrix1_size))
        print('output shape: ', out_shape)

        # print(csdl.einsum(
        #             x,
        #             np.ones(tuple(matrix1_size)),
        #             action=f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}'
        #         ))
        return 1
        if len(input_dp_index) <= 1:
            return csdl.reshape(
                csdl.einsum(
                    x,
                    np.ones(tuple(matrix1_size)),
                    action=f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}'
                ),
                out_shape,
                )
        else:
            in_shape = list(x.shape)
            #print('original shape: ', in_shape)
            in_shape.pop(0)
            insert_indx = 0
            for j in range(len(input_dp_index)):
                in_shape.insert(insert_indx, input_n_1d_qp[j])
                insert_indx += 1
            in_shape = tuple(in_shape)
            #print('new shape: ', in_shape)
            return csdl.reshape(csdl.einsum(f'{einsum_input_string1},{einsum_input_string2}->{einsum_output_string}', np.reshape(x, in_shape), np.ones(tuple(matrix1_size))), out_shape)
    return expansion_func