from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.src.graph.variable import Variable
from typing import Union
from csdl_alpha.src.operations.loops.loop import IterationVariable
from csdl_alpha.src.operations.loops.new_loop.utils import build_iteration_variables
from csdl_alpha.src.operations.loops.new_loop.feedbacks import Feedback, Feedbacks

class LoopBuilder:
    def __init__(
            self,
            loop_graph:Graph,
            iter_vars:dict['IterationVariable', list[int]],
            ) -> None:
        self.loop_graph:Graph = loop_graph
        self.iters:dict[IterationVariable,list[int]] = iter_vars
        self.length:int = len(list(iter_vars.values())[0])

        # Keep track of loop feedbacks
        self.feedbacks:Feedbacks = Feedbacks()

        # Keep track of loop inputs and outputs
        # By the end, all loop graph variables involved with feedbacks, stacks and accrues
        # plus any other specifed outputs will be added to outputs
        self.outputs:dict[Variable,Variable] = {}
        self.inputs:dict[Variable,Variable] = {}

        # Special outputs where the values are not direct outputs of the loop
        self.accrued:dict[Variable,Variable] = {}
        self.stacked:dict[Variable,Variable] = {}

        # If locked, can only specify information about outer graph like stacked variables
        self.locked:bool = False

    def check_locked(self):
        if not self.locked:
            raise ValueError("This method can only be called after the builder is locked (no more changes to the loop body). Call lock() to lock the builder.")

    def get_loop_indices(self)->Union['IterationVariable',tuple['IterationVariable']]:

        # Return loop iteration variables in the order they are defined from self.iters
        iter_vars = []
        for iter_var, iter_vals in self.iters.items():
            iter_vars.append(iter_var)

        if len(iter_vars) == 1:
            return iter_vars[0]
        else:
            return tuple(iter_vars)
    
    def initialize_feedback(self, ext_input_var:Variable)->Variable:
        """Used to build loop feedbacks in conjunction with finalize_feedback

        Parameters
        ----------
        ext_input_var : Variable
            The initial condition of a feedback variable. For example:

        Returns
        -------
        Variable
            The internal input variable of the feedback used for all subsequent iterations. 

        Examples
        --------

        # Without using the loop builder:
        x = Variable()
        for i in range(10):
            x = x+1

        # Using the loop builder:
        x0 = Variable()
        with enter_loop(10) as loop_builder:
            x = loop_builder.initialize_feedback(x0)
            x_new = x+1
            loop_builder.finalize(x, x_new)
        """
        return self.feedbacks.initialize_feedback(ext_input_var)

    def finalize_feedback(
            self,
            int_input_var:Variable,
            output:Variable,
            )->None:
        """Used to build loop feedbacks in conjunction with initialize_feedback

        Parameters
        ----------
        int_input_var : Variable
            The output of a previously called initialize_feedback 
        output : Variable
            The output of the feedback
        """
        return self.feedbacks.finalize_feedback(int_input_var, output)

    def add_output(self, output:Variable)->Variable:
        self.outputs[output] = {}

        if output not in self.loop_graph.node_table:
            raise ValueError(f"Output {output} not found in loop graph.")
        return output
    
    def add_input(self, input:Variable)->None:
        self.check_locked()
        self.inputs[input] = {}
        if input not in self.loop_graph.node_table:
            raise ValueError(f"Input {input} not found in loop graph.")

    def add_pure_accrue(self, accrue_target:Variable)->Variable:
        """ Specify a variable in the loop body to be accrued as an output.

        Parameters
        ----------
        accrue_target : Variable
            A variable in the loop body

        Returns
        -------
        Variable
            A variable that is the sum of all values of the accrue target over all iterations of the loop

        Raises
        ------
        ValueError
            If the accrue target is not found in the loop
        """
        self.check_locked()
        if accrue_target not in self.loop_graph.node_table:
            raise ValueError(f"Accrue target {accrue_target} not found in loop graph.")
        if accrue_target in self.accrued:
            return self.accrued[accrue_target]
        else:
            accrued_var = Variable(name = f'{accrue_target.name}_stacked', shape = accrue_target.shape)
            self.accrued[accrue_target] = accrued_var
            return accrued_var
    
    def add_stack(self, stack_target:Variable)->Variable:
        """Specify a variable IN the loop body to be stacked as an output.

        Parameters
        ----------
        stack_target : Variable
            A variable in the loop body

        Returns
        -------
        Variable
            A stacked version of the variable with shape (num iter, *stack_target.shape)

        Raises
        ------
        ValueError
            If the stack target is not found in the loop graph
        """
        self.check_locked()
        if stack_target not in self.loop_graph.node_table:
            raise ValueError(f"Stack target {stack_target} not found in loop graph.")
        if stack_target in self.stacked:
            return self.stacked[stack_target]
        else:
            stacked_var = Variable(name = f'{stack_target.name}_stacked',shape = (self.length,) + stack_target.shape)
            self.stacked[stack_target] = stacked_var
            return stacked_var
    
    def lock(self):
        """Called after all loop body operations are specified. This locks the builder and only allows changes to the outer graph.
        """
        self.feedbacks.check()
        self.locked = True

    def finalize(
            self,
            add_all_outputs:bool = True,
        ):
        self.check_locked()

        # add all intermediate variables as outputs if specified
        if add_all_outputs:
            from csdl_alpha.src.graph.operation import Operation
            for node in self.loop_graph.node_table.keys():
                if isinstance(node, Operation):
                    for output in node.outputs:
                        self.add_output(output)

        # add all inputs:
        for input_var in self.loop_graph.inputs:
            self.add_input(input_var)

        # Stack all feedback variables if not already stacked
        for feedback in self.feedbacks._int_input_to_feedback.values():
            self.add_stack(feedback.internal_input)
            self.add_output(feedback.output)

        # Build the actual operation object
        from csdl_alpha.src.operations.loops.new_loop.new_loop import NewLoop
        loop = NewLoop(
            loop_builder = self,
        )
        return loop.finalize_and_return_outputs()

    def __repr__(self) -> str:
        op_id = super().__repr__()

        string = f"\nLoopBuilder ({op_id}) with {self.length} iterations.\n"
        string += f"\t{len(self.loop_graph.node_table)} Body graph nodes\n"
        string += f"\t{len(self.inputs)} Input(s)\n"
        string += f"\t{len(self.outputs)} Standard output(s)\n"
        string += f"\t{len(self.feedbacks._int_input_to_feedback)} Feedbacks(s)\n"
        string += f"\t{len(self.accrued)} Accrued Output(s)\n"
        string += f"\t{len(self.stacked)} Stacked Output(s)\n"
        return string

class enter_loop(object):
    def __init__(self, vals:list[list[int]]):
        self.loop_builder = None
        self.vals = vals
    
    def __enter__(self) -> LoopBuilder:
        import csdl_alpha as csdl
        self.recorder = csdl.get_current_recorder() 
        
        self.recorder._enter_subgraph(
            add_missing_variables=True,
        )
        self.loop_graph = self.recorder.active_graph
        self.loop_builder = LoopBuilder(
            loop_graph = self.loop_graph,
            iter_vars = build_iteration_variables(self.vals),
        )
        return self.loop_builder

    def __exit__(self, *args):
        # handle any exceptions
        if any(args):
            return False

        self.loop_builder.lock()
        self.recorder._exit_subgraph()
