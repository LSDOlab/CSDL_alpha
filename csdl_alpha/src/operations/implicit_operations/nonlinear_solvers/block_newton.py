from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import (
    NonlinearSolver,
    check_variable_shape_compatibility,
    check_run_time_tolerance,
)
from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.newton import Newton
from csdl_alpha.src.graph.variable import Variable, ImplicitVariable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np
import rustworkx as rx
from typing import Union, Callable, Optional


class BlockNewton(Newton):
    """
    Block-sparse Newton solver with automatic structural reachability detection,
    KKT block-symmetry exploitation, and primal Hessian symmetry.

    This solver accelerates Newton iteration by:
    1. Identifying structurally decoupled state-residual blocks via Rustworkx DAG reachability,
       instantiating exact zero constant blocks with zero autodiff sweeps.
    2. Exploiting KKT transpose symmetry (e.g. d(r_primal)/d(lambda) = (d(c)/d(primal))^T),
       eliminating expensive reverse AD sweeps through complex mechanics/physics graphs.
    3. Exploiting primal Hessian symmetry (d(r_pi)/d(pj) = (d(r_pj)/d(pi))^T), halving
       off-diagonal second derivative evaluations between primal states.
    4. Supporting constant/analytical blocks (e.g. exact quadratic objective Hessians) to bypass
       autodiff sweeps entirely for designated blocks.
    5. Row-grouping active derivative evaluations per residual to compute all coupled state blocks
       in a single reverse-mode pass.
    """

    def __init__(
        self,
        name: str = 'block_newton_nlsolver',
        print_status: bool = True,
        tolerance: float = 1e-10,
        max_iter: int = 100,
        elementwise_states: bool = False,
        residual_jac_kwargs: Optional[dict[str, Union[bool, int, float, str]]] = None,
        symmetric_primal: bool = False,
    ):
        super().__init__(
            name=name,
            print_status=print_status,
            tolerance=tolerance,
            max_iter=max_iter,
            elementwise_states=elementwise_states,
            residual_jac_kwargs=residual_jac_kwargs,
        )
        self._zero_blocks: set[tuple[Variable, Variable]] = set()
        self._constant_blocks: dict[tuple[Variable, Variable], Variable] = {}
        self._transpose_blocks: dict[tuple[Variable, Variable], tuple[Variable, Variable, float]] = {}
        self._kkt_constraints: list[tuple[Variable, Variable]] = []
        self.symmetric_primal: bool = symmetric_primal

    def add_kkt_constraint(
        self,
        multiplier: Variable,
        constraint: Variable,
        initial_value: Optional[Union[Variable, float, int]] = None,
        tolerance: Optional[Union[Variable, float, int]] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Register a KKT constraint multiplier and constraint residual pair.

        Automatically establishes:
        - Zero block for d(c)/d(lambda) = 0 across all constraint multipliers.
        - Transpose symmetry: d(r_primal)/d(lambda) = scale * (d(c)/d(primal))^T for all
          primal states influencing the constraint.
        - Structural zeros for all primal states decoupled from the constraint.

        Parameters
        ----------
        multiplier : Variable
            The Lagrange multiplier state variable.
        constraint : Variable
            The constraint residual variable c(x) = 0.
        initial_value : Optional[Union[Variable, float, int]]
            Initial value for the multiplier.
        tolerance : Optional[Union[Variable, float, int]]
            Convergence tolerance for the constraint residual.
        scale : float
            Scaling factor applied to the transpose block (e.g. 1 - alpha_f in Generalized-Alpha).
        """
        self.add_state(multiplier, constraint, initial_value=initial_value, tolerance=tolerance)
        self._kkt_constraints.append((multiplier, constraint, scale))

    def set_block_zero(self, of: Variable, wrt: Variable) -> None:
        """
        Explicitly designate a Jacobian block d(of)/d(wrt) as identically zero.

        Parameters
        ----------
        of : Variable
            The residual variable.
        wrt : Variable
            The state variable with respect to which the derivative is zero.
        """
        self._zero_blocks.add((of, wrt))

    def set_block_constant(
        self,
        of: Variable,
        wrt: Variable,
        value: Union[Variable, np.ndarray, float, int],
    ) -> None:
        """
        Explicitly designate a Jacobian block d(of)/d(wrt) as a known constant or precomputed matrix.

        Parameters
        ----------
        of : Variable
            The residual variable.
        wrt : Variable
            The state variable.
        value : Union[Variable, np.ndarray, float, int]
            The constant matrix or scalar/diagonal value.
        """
        import csdl_alpha as csdl

        expected_shape = (of.size, wrt.size)
        if isinstance(value, (int, float)):
            if expected_shape == (1, 1):
                mat_val = np.array([[float(value)]])
            elif of.size == wrt.size:
                mat_val = float(value) * np.eye(of.size)
            else:
                mat_val = np.full(expected_shape, float(value))
            var = csdl.Variable(name=f"const_{of.name}_wrt_{wrt.name}", value=mat_val)
        elif isinstance(value, np.ndarray):
            if value.shape != expected_shape:
                if value.size == 1:
                    if of.size == wrt.size:
                        mat_val = float(value.item()) * np.eye(of.size)
                    else:
                        mat_val = np.full(expected_shape, float(value.item()))
                elif value.ndim == 1 and value.size == of.size and of.size == wrt.size:
                    mat_val = np.diag(value)
                else:
                    raise ValueError(
                        f"Constant block shape mismatch for d({of.name})/d({wrt.name}): "
                        f"expected {expected_shape}, got {value.shape}"
                    )
            else:
                mat_val = value
            var = csdl.Variable(name=f"const_{of.name}_wrt_{wrt.name}", value=mat_val)
        elif isinstance(value, Variable):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Constant block shape mismatch for d({of.name})/d({wrt.name}): "
                    f"expected {expected_shape}, got {value.shape}"
                )
            var = value
        else:
            raise TypeError(f"Unsupported constant value type: {type(value)}")

        self._constant_blocks[(of, wrt)] = var

    def set_block_transpose(
        self,
        of: Variable,
        wrt: Variable,
        transpose_of: Variable,
        transpose_wrt: Variable,
        scale: float = 1.0,
    ) -> None:
        """
        Designate that block d(of)/d(wrt) is the transpose of d(transpose_of)/d(transpose_wrt),
        optionally multiplied by a scalar factor.

        Parameters
        ----------
        of : Variable
            The residual variable for the transposed block.
        wrt : Variable
            The state variable for the transposed block.
        transpose_of : Variable
            The source residual variable.
        transpose_wrt : Variable
            The source state variable.
        scale : float
            Optional scalar multiplier (default 1.0).
        """
        self._transpose_blocks[(of, wrt)] = (transpose_of, transpose_wrt, float(scale))

    def _build_full_residual_jacobian(
        self,
        for_deriv: bool,
    ) -> Variable:
        """
        Constructs the full block-sparse Jacobian matrix leveraging graph reachability
        and transpose symmetry.
        """
        import csdl_alpha as csdl

        states = list(self.state_to_residual_map.keys())
        residuals = list(self.state_to_residual_map.values())

        recorder = csdl.get_current_recorder()
        active_graph = (
            self.residual_graph
            if (for_deriv and self.residual_graph is not None)
            else recorder.active_graph
        )

        # Precompute structural ancestors for each residual in the computation graph
        ancestors_map: dict[Variable, set[int]] = {}
        for r in residuals:
            if r in active_graph.node_table:
                ancestors_map[r] = set(
                    rx.ancestors(active_graph.rxgraph, active_graph.node_table[r])
                )
            else:
                ancestors_map[r] = set()

        # Automatically configure KKT constraint blocks
        kkt_mults = {item[0] for item in self._kkt_constraints}
        primal_states = [s for s in states if s not in kkt_mults]

        for item in self._kkt_constraints:
            mult, c_res = item[0], item[1]
            c_scale = float(item[2]) if len(item) > 2 else 1.0
            # Multiplier block in constraint row is zero: d(c)/d(lambda) = 0
            for other_mult in kkt_mults:
                self._zero_blocks.add((c_res, other_mult))

            # Transpose symmetry: d(r_primal)/d(lambda) = scale * (d(c)/d(primal))^T
            for p_state in primal_states:
                p_res = self.state_to_residual_map[p_state]
                p_state_node = active_graph.node_table.get(p_state)
                if p_state_node is not None and p_state_node in ancestors_map[c_res]:
                    self._transpose_blocks[(p_res, mult)] = (c_res, p_state, c_scale)
                else:
                    self._zero_blocks.add((c_res, p_state))
                    self._zero_blocks.add((p_res, mult))

        # Automatically exploit primal-primal Hessian symmetry: d(r_pj)/d(pi) = (d(r_pi)/d(pj))^T
        if self.symmetric_primal:
            for i, p_state_i in enumerate(primal_states):
                p_res_i = self.state_to_residual_map[p_state_i]
                p_i_node = active_graph.node_table.get(p_state_i)
                for j in range(i + 1, len(primal_states)):
                    p_state_j = primal_states[j]
                    p_res_j = self.state_to_residual_map[p_state_j]
                    p_j_node = active_graph.node_table.get(p_state_j)

                    pair_ji = (p_res_j, p_state_i)
                    pair_ij = (p_res_i, p_state_j)
                    if (
                        pair_ji not in self._transpose_blocks
                        and pair_ji not in self._zero_blocks
                        and pair_ji not in self._constant_blocks
                        and pair_ij not in self._transpose_blocks
                        and pair_ij not in self._zero_blocks
                        and pair_ij not in self._constant_blocks
                    ):
                        in_j = p_i_node is not None and p_i_node in ancestors_map[p_res_j]
                        in_i = p_j_node is not None and p_j_node in ancestors_map[p_res_i]
                        if in_j and in_i:
                            self._transpose_blocks[pair_ji] = (p_res_i, p_state_j, 1.0)
                        elif not in_j and not in_i:
                            self._zero_blocks.add(pair_ji)
                            self._zero_blocks.add(pair_ij)

        J_blocks: dict[tuple[Variable, Variable], Variable] = {}

        # Phase 1: Determine structural zero blocks, constant blocks, and active autodiff blocks per residual row
        for r in residuals:
            active_wrts: list[Variable] = []
            for s in states:
                if (r, s) in self._zero_blocks:
                    J_blocks[(r, s)] = csdl.Variable(
                        name=f"zero_{r.name}_wrt_{s.name}",
                        value=np.zeros((r.size, s.size)),
                    )
                elif (r, s) in self._constant_blocks:
                    J_blocks[(r, s)] = self._constant_blocks[(r, s)]
                elif (r, s) in self._transpose_blocks:
                    # Postponed to Phase 2
                    pass
                else:
                    s_node = active_graph.node_table.get(s)
                    if s_node is None or s_node not in ancestors_map[r]:
                        # Structurally disconnected in DAG -> exact zero block
                        J_blocks[(r, s)] = csdl.Variable(
                            name=f"sparse_zero_{r.name}_wrt_{s.name}",
                            value=np.zeros((r.size, s.size)),
                        )
                    else:
                        active_wrts.append(s)

            # Compute all active blocks for residual r in a single reverse-mode pass
            if len(active_wrts) > 0:
                deriv_kwargs = self.residual_jac_kwargs.copy()
                deriv_kwargs["as_block"] = False
                if for_deriv and self.residual_graph is not None:
                    deriv_kwargs["graph"] = self.residual_graph

                derivs = csdl.derivative(r, active_wrts, **deriv_kwargs)
                for s in active_wrts:
                    J_blocks[(r, s)] = derivs[s]

        # Phase 2: Resolve transposed blocks
        for (r, s), (tr, ts, scale) in self._transpose_blocks.items():
            if (tr, ts) not in J_blocks:
                raise KeyError(
                    f"Target block ({tr.name}, {ts.name}) for transpose block ({r.name}, {s.name}) was not computed."
                )
            target_block = J_blocks[(tr, ts)]
            # Check shape compatibility
            expected_shape = (r.size, s.size)
            transposed_shape = (
                target_block.shape[1] if len(target_block.shape) > 1 else 1,
                target_block.shape[0],
            )
            if transposed_shape != expected_shape:
                raise ValueError(
                    f"Transpose block dimension mismatch: block ({r.name}, {s.name}) expected shape "
                    f"{expected_shape}, but transposed block ({tr.name}, {ts.name}).T has shape {transposed_shape}"
                )
            tr_block = target_block.T()
            if scale != 1.0:
                tr_block = tr_block * scale
            J_blocks[(r, s)] = tr_block

        # Phase 3: Validate completeness and assemble block matrix
        block_rows: list[list[Variable]] = []
        for r in residuals:
            row: list[Variable] = []
            for s in states:
                if (r, s) not in J_blocks:
                    raise KeyError(f"Block ({r.name}, {s.name}) was not populated during block Jacobian assembly.")
                row.append(J_blocks[(r, s)])
            block_rows.append(row)

        from csdl_alpha.src.operations.linalg.blockmat import blockmat

        jac = blockmat(block_rows)
        jac.add_name(f"{self.name}_jac")
        return jac
