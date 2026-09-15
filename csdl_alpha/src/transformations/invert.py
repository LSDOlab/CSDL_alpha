from csdl_alpha.src.transformations.transformation import TransformationBase
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import get_type_string, variablize

import networkx as nx
import rustworkx as rx
import numpy as np
import string

from typing import Any, Union
from csdl_alpha.utils.inputs import ingest_value, get_shape, process_shape_and_value, get_type_string
from typing import Optional

def find_next_node(invertible_inputs:list[Variable], available_nodes:set[Variable])->Variable:
    if available_nodes is None:
        new_lhs = invertible_inputs[0] # Just take random invertible input for now
        return new_lhs
    else:
        # If a target is specified, try to invert towards that target
        for var in invertible_inputs:
            if var in available_nodes:
                return var

def find_available_nodes(target:Optional[Variable], graph:Graph)->set[Variable]:
    if target is None:
        return None
    else:
        descendents = graph.descendants(target, include_node=True)
        return descendents

class EqualityInversion(TransformationBase):

    def apply(
            self,
            lhs:Variable,
            rhs:Union[np.ndarray, Variable],
            target:Optional[Variable] = None,
            aux_info:bool = False,
            debug:bool=True)->tuple[Variable, Variable]:
        recorder = self.get_current_recorder()
        graph = recorder.active_graph
        import csdl_alpha as csdl
        ct = csdl.transformation_helper

        lhs = variablize(lhs)
        rhs = variablize(rhs)
        if lhs.shape != rhs.shape: raise ValueError(f"lhs and rhs must have the same shape to be equated. Got {lhs.shape} and {rhs.shape}.")
        
        # Start backwards from lhs and flip inverses to the rhs
        # lhs = f(f(f(...f(x)))) = c = rhs
        #         f(f(...f(x)))) = f^-1(c)
        #           f(...f(x)))) = f^-1(f^-1(c))
        # and so on until we get a function that represents the inverse of the chain
        # If a function is not invertible, stop and return both sides

        # At a given node, we choose an input to invert for which we call
        # y_0, y_1, ..., y_M = f(x_0, x_1, ..., x_N)
        # If we want to invert for x_i:
        # x_i = f^-1(y_0, y_1, ..., y_M, x_0, ..., x_N)

        # We keep track of lhs and rhs where:
        # - lhs is the variable in the ORIGINAL forward graph
        # - rhs is the variable in the NEW inverted graph that we are pushing inversions through
        """
           ::::::ORIGINAL::::::
                 (LHS old)
           y0  y1  y2
           o   o   o
           ^   ^   ^
           |   |   |
               o f
           ^   ^   ^
           |   |   |
           o   o   o
           x0  x1  x2
             (LHS
              new)

          ::::::INVERTED::::::
           (RHS new)
               o 
               ^
               |
               o f^-1
           ^   ^   ^   ^    ^
           |   |   |   |    |
           o   o   o   o    o 
           x0  x2  y0  y1  (RHS old) 
        """

        # root_rhs:Variable = csdl.Variable(shape=rhs.shape) # potentially change later?
        root_rhs:Variable = rhs
        new_lhs:Variable = lhs
        new_rhs:Variable = root_rhs
        inverted_chain:list[Operation] = []
        iter:int = 0

        available_nodes = find_available_nodes(target, graph)

        while True:
            assert isinstance(new_lhs, Variable)
            predecessor_ops = ct.predecessors(new_lhs, graph)
            if len(predecessor_ops) == 0: break # This variable is an input variable
            predecessor_op:Operation = predecessor_ops[0]
            assert isinstance(predecessor_op, Operation)

            # Checks: 
            # 1) Is this operation invertible?
            invertible_inputs = predecessor_op.get_invertible_inputs()
            if invertible_inputs is None:
                if debug:
                    print(f'{iter}: No op inversion for {predecessor_op.name}...')
                break
            # 2) Can we invert towards the desired target (if specified)?
            candidate = find_next_node(invertible_inputs, available_nodes)
            if candidate is None:
                if debug:
                    print(f'{iter}: Cannot invert toward target any further (operation {predecessor_op.name})')
                break
            
            # At this point, we know we can invert this operation
            inverted_chain.append(predecessor_op)
            if debug:
                print(f'{iter}: Inverting op {predecessor_op.name}...')

            old_lhs = new_lhs
            new_lhs = candidate

            # Sometimes shapes get messed up during inverses, so we need to reshape here
            if new_rhs.shape != old_lhs.shape: 
                if debug: print(f"Target and new shapes don't match ({new_rhs.shape} vs {old_lhs.shape}), reshaping...")
                new_rhs = new_rhs.reshape(old_lhs.shape)
            
            new_rhs = predecessor_op.inverse(
                x_target = new_lhs,
                y_target = old_lhs,
                y_value = new_rhs,
                debug=debug,
            )#.add_name(f"inverse_{iter}")

            # new_rhs.name = f"inverse_RHS_{iter}"
            # new_lhs.name = f"inverse_LHS_{iter}"

            iter += 1

        if debug:
            print(f"Inverted operations: ", end="")
            for op in inverted_chain:
                print(f"{op.name}->", end="")
            print()

        # Maybe in the future, Create subgraph operation from root_rhs to new_rhs
        new_lhs.add_name("new_lhs")
        new_rhs.add_name("new_rhs")
        if aux_info:
            return new_lhs, new_rhs, inverted_chain
        else:
            return new_lhs, new_rhs

    def transform(
            self,
            lhs:Variable,
            rhs:Union[np.ndarray, Variable],
            target:Optional[Variable] = None,
            aux_info:bool = False,
            debug:bool=True
        )->tuple[Variable, Variable]:
        """Performs the inversion transformation on an equality of the form lhs = rhs by 
        starting backwards from lhs and flipping inverses to the rhs
        
        lhs = f(f(f(...f(x)))) = rhs
                f(f(...f(x)))) = f^-1(rhs)
                  f(...f(x)))) = f^-1(f^-1(rhs))

        Parameters
        ----------
        lhs : Variable
            Variable to be inverted
        rhs : Union[np.ndarray, Variable]
            Variable that lhs is supposed to be equal to
        target : Optional[Variable], optional
            Variable to invert towards, by default None
            If None, the transformation will invert in a random direction along the chain of operations
        aux_info : bool, optional
            If True, returns the chain of inverted operations as well, by default False
        debug : bool, optional
            If True, prints debug information during the transformation, by default True

        Returns
        -------
        tuple[Variable, Variable]
            Transformed equivalent lhs and rhs variables. If aux_info is True, also returns the chain of inverted operations.
        """

        return super().transform(
            lhs,
            rhs,
            target=target,
            aux_info=aux_info,
            debug=debug,
        )
