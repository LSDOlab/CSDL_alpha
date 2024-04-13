from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike
import numpy as np
import math

from typing import Union

class BracketedSearch(NonlinearSolver):
    
    def __init__(
                self,
                name:str = 'bracketed_search',
                print_status:bool = True,
                tolerance:VariableLike=1e-10,
                max_iter:int=100,
            ):
            """Initialize the BracketedSearch solver.

            Parameters
            ----------
            name : str, optional
                The name of the solver. Defaults to 'bracketed_search'.
            print_status : bool, optional
                Whether to print the solver status during execution. Defaults to True.
            tolerance : VariableLike, optional
                The tolerance value used to determine convergence. Defaults to 1e-10.
            max_iter : int, optional
                The maximum number of iterations allowed. Defaults to 100.
            """

            super().__init__(
                name = name,
                print_status = print_status,
                tolerance = tolerance,
                max_iter = max_iter
            )

            self.add_metadata('tolerance', tolerance)
            self.add_metadata('max_iter', max_iter)

    def add_state(
            self,
            state: Variable,
            residual: Variable,
            bracket: tuple[VariableLike, VariableLike],
            tolerance: VariableLike = None
        ):
        """Add a state to the solver.

        This method adds a state to the solver along with its corresponding residual. It also allows specifying a bracket
        and tolerance for the state.

        Parameters
        ----------
        state : Variable
            The state variable to be added to the solver.
        residual : Variable
            The residual variable corresponding to the state.
        bracket : tuple[Union[Variable, np.ndarray, float, int], Union[Variable, np.ndarray, float, int]]
            The bracket for the state. It should be a tuple containing two elements, each of which can be a Variable,
            np.ndarray, float, or int. Must have a shape of (1,) or match the shape of the state.
        tolerance : Union[Variable, np.ndarray, float], optional
            The tolerance for the state. It can be a Variable, np.ndarray, float, or int. If not provided, the tolerance
            from the solver's metadata will be used. Must have a shape of (1,) or match the shape of the state.

        Raises
        ------
        ValueError
            If the bracket is not a tuple or does not have two elements.
        ValueError
            If the bracket elements have invalid types or shapes.
        ValueError
            If the tolerance has an invalid type or shape.
        """

        self.add_state_residual_pair(state, residual)

        # check if bracket is valid and store it
        if not isinstance(bracket, tuple):
            raise ValueError("Bracket must be a tuple")
        if len(bracket) != 2:    
            raise ValueError("Bracket must have two elements")
        for element in bracket:
            if isinstance(element, (Variable, np.ndarray)):
                if not (element.shape == (1,) or element.shape == state.shape):
                    raise ValueError("Bracket shape must match state shape")
            elif not isinstance(element, (float, int)):
                raise ValueError(f"Invalid bracket element type, got {type(element)}")
        self.add_state_metadata(state, 'bracket', bracket)

        # check if tolerance is valid and store it
        if tolerance is not None:
            if isinstance(tolerance, Variable, np.ndarray):
                if not (tolerance.shape == (1,) or tolerance.shape == state.shape):
                    raise ValueError(f"Tolerance shape must be scalar or match state shape. {tolerance.shape} given")
            elif not isinstance(tolerance, (float, int)):
                raise ValueError(f"Invalid tolerance type, got {type(tolerance)}")
            self.add_state_metadata(state, 'tolerance', tolerance)
        else:
            self.add_state_metadata(state, 'tolerance', self.metadata['tolerance'])

    def _inline_solve_(self):
        iter = 0

        # I'm going to assemble everything into a big vector, solve it, and then unpack it

        # First, I need to know the size of the vector, and the indices of each state
        indices_dict = {}
        size = 0
        for state in self.state_to_residual_map.keys():
            state_size = math.prod(state.shape)
            indices_dict[state] = slice(size, size+state_size)
            size += state_size

        x_upper = np.empty(size)
        x_lower = np.empty(size)
        r_lower = np.empty(size)
        r_update = np.empty(size)
        tolerance = np.empty(size)

        def compute(x, r):
            for state, index in indices_dict.items():
                state.value = x[index].reshape(state.shape)
            self.update_residual()
            for state, index in indices_dict.items():
                r[index] = self.state_to_residual_map[state].value.flatten()

        def to_array(x):
            if isinstance(x, Variable):
                return x.value
            elif isinstance(x, (float, int)):
                return np.array([x])

        # Now, I need to populate the x vectors with the initial bracket values
        # and populate the tolerance vector
        for state in self.state_to_residual_map.keys():
            bracket = self.state_metadata[state]['bracket']
            ith_tolerance = self.state_metadata[state]['tolerance']
            # bracket indices could be Variable or ndarray of shape (1,) or state.shape
            # or float or int
            # I want to turn them all into numpy arrays
            
            # now I can flatten them and put them in the x vectors
            x_lower[indices_dict[state]] = to_array(bracket[0]).flatten()
            x_upper[indices_dict[state]] = to_array(bracket[1]).flatten()
            tolerance[indices_dict[state]] = to_array(ith_tolerance).flatten()

        # Now, I need to populate the r vectors with the residuals of the initial bracket values
        # I'm also going to record whether the sign of the lower is negative (True) or positive (False)
        compute(x_lower, r_lower)
        # compute(x_upper, r_upper)
        r_sign = np.any(r_lower < 0, axis=0)

        # Now, I need to iterate until the bracket is small enough
        while True:
            # compute the midpoint
            x_mid = (x_upper + x_lower) / 2
            compute(x_mid, r_update)

            # check if the midpoint is the new upper or lower bound
            lower_mask = np.any(r_update < 0 and r_sign, axis=0)

            x_lower[lower_mask] = x_mid[lower_mask]
            x_upper[~lower_mask] = x_mid[~lower_mask]

            # check if the bracket is small enough
            if np.all(np.abs(x_upper - x_lower) < tolerance):
                converged = True
                break

            # check if we've hit the max iterations
            iter += 1
            if iter > self.metadata['max_iter']:
                converged = False
                break

        if self.print_status:
            print(self._inline_print_nl_status(iter, converged))


        
