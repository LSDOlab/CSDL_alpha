from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
from csdl_alpha.src.graph.graph import _copy_to_current_graph

from csdl_alpha.src.operations.mpi.operatives import mpi_sum
from csdl_alpha.utils.inputs import variablize

from typing import Union
import numpy as np
import csdl_alpha as csdl

class MPIRegionOperation(SubgraphOperation):
    def __init__(self, mpi_region: 'MPIRegion', name: str = "mpi_region") -> None:
        self.mpi_region = mpi_region
        self.name = name

        self.global_outputs = list(mpi_region.global_outputs.keys())
        recorder = csdl.get_current_recorder()
        for output in self.global_outputs:
            recorder._add_node(output)

        self.global_inputs = list(mpi_region.global_inputs.keys())

        super().__init__(*self.global_inputs)
        self.set_outputs(self.global_outputs)
        self.assign_subgraph(mpi_region.mpi_region_graph)

    def compute_inline(self, *args):
        self.get_subgraph().execute_inline()
        if len(self.outputs) == 1:
            return self.outputs[0].value
        else:
            return [output.value for output in self.outputs]
        
    def compute_jax(self, *args):
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        jax_fn = create_jax_function(self.get_subgraph(), self.outputs, self.inputs)
        return tuple(jax_fn(*args))

    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:self.num_inputs]
        outputs = inputs_and_outputs[self.num_inputs:]

        cotangents_map = {}
        # seeds = {global_out: cotangents[global_out] for _, global_out in zip(self.outputs, outputs)}
        seeds = {}
        for _, global_out in zip(self.outputs, outputs):
            if cotangents.check(global_out):
                seeds[global_out] = cotangents[global_out]

        def direct_passthough(x):
            return x
        def mpi_sum_comm(x):
            return mpi_sum(x, self.mpi_region.comm)

        wrts, wrt_accumulate = [], {}
        for i, (input_var_orig, input_var) in enumerate(zip(self.inputs, inputs)):
            if self.mpi_region.global_inputs[input_var_orig] == 'split':
                accumulate_func = direct_passthough
            else:
                accumulate_func = mpi_sum_comm
            wrts.append(input_var)
            wrt_accumulate[input_var] = accumulate_func

        merged_seeds = {}
        from csdl_alpha.src.operations.derivatives.reverse import vjp
        with csdl.experimental.mpi.enter_mpi_region(self.mpi_region.rank, self.mpi_region.comm, name=self.name+'_vjp') as region:

            _copy_to_current_graph(
                self.get_subgraph(),
                {},
                add_to_graph_inputs = True)

            vjps = vjp([(var,seed) for var,seed in seeds.items()], wrts, self.get_subgraph())
            for wrt, vjp_propagated in vjps.items():
                if vjp_propagated is not None:
                    merged_seeds[wrt] = region.merge_custom(vjp_propagated, merge_func=wrt_accumulate[wrt])
            # region.mpi_region_graph.visualize()

        for _, input_var in zip(self.inputs, inputs):
            if cotangents.check(input_var):
                if input_var in merged_seeds:
                    cotangents.accumulate(input_var, merged_seeds[input_var])
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical', filename='fwd')
        # exit()
        
class MPIRegion():
    def __init__(self, mpi_region_graph:Graph, rank:int, comm, name:str ) -> None:
        self.mpi_region_graph = mpi_region_graph
        self.rank = rank
        self.comm = comm
        self.name = name
        self.global_outputs = {}
        self.global_inputs = {}

    def set_as_global_output(self, output:Variable) -> Variable:
        """
        Marks a variable as a global output of the MPI region.
        """
        if output in self.global_outputs:
            raise ValueError(f"Variable {output.info()} is already a global output.")
        self.global_outputs[output] = None
        return output

    def split_custom(self, x: Variable, split_func:callable) -> Variable:
        """
        Splits the variables args using the provided split function.
        """
        assert isinstance(x, Variable), "Split function must take a csdl variable as input"
        if x in self.mpi_region_graph.node_table:
            raise ValueError(f"Only global variables can be split, {x.info()} is already in MPI region")
        self.global_inputs[x] = 'split'
        return split_func(x)
        
    def split_constant(self, x: np.ndarray) -> Variable:
        assert isinstance(x, np.ndarray), "Split must be a numpy array"
        return variablize(x)

    def merge_custom(self, x: Variable, merge_func:callable) -> Variable:
        """
        Merges the variables args using the provided merge function.
        """
        global_output = merge_func(x)
        if not isinstance(global_output, Variable):
            raise ValueError("Merge function must return a csdl variable!")
        self.set_as_global_output(global_output)
        return global_output

    def finalize(self):
        for input_var in self.mpi_region_graph.inputs:
            if input_var not in self.global_inputs:
                self.global_inputs[input_var] = 'global'

        from csdl_alpha.src.graph.operation import Operation
        mpi_region_op = MPIRegionOperation(
            mpi_region = self,
            name = self.name,
        )
        mpi_region_op.finalize_and_return_outputs()

class enter_mpi_region(object):
    def __init__(self, rank:int, comm, name:str = "mpi_region"):
        self.rank = rank
        self.comm = comm
        self.name = name

    def __enter__(self):
        import csdl_alpha as csdl
        self.recorder = csdl.get_current_recorder() 
        
        self.recorder._enter_subgraph(
            add_missing_variables=True,
        )
        self.mpi_region_graph = self.recorder.active_graph
        self.mpi_region = MPIRegion(
            mpi_region_graph = self.mpi_region_graph,
            rank = self.rank,
            comm = self.comm,
            name = self.name
        )
        return self.mpi_region
    
    def __exit__(self, *args):

        # handle any exceptions
        if any(args):
            return False
        self.recorder._exit_subgraph()
        
        self.mpi_region.finalize()

