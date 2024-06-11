from csdl_alpha.backends.simulator import SimulatorBase, Recorder
from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
from csdl_alpha.src.graph.variable import Variable

class JaxSimulator(SimulatorBase):

    def __init__(
            self, 
            recorder:Recorder,
            ):
        super().__init__(recorder)
        self.recorder:Recorder = recorder
        self.initialize_totals = False

        self.hash = None

    def run(self):
        import jax.numpy as jnp
        jnp_inputs = []
        for input in self.input_vars:

            if input.value is None:
                raise ValueError(f"Input {input} has no value set.")
            
            jnp_inputs.append(jnp.array(input.value))

        jnp_inputs = [var.value for var in self.input_vars]
        outputs = self.jax_function(*jnp_inputs)

        for i, output in enumerate(self.output_vars):
            output.value = outputs[i]