import csdl_alpha as csdl
from csdl_alpha.src.transformations.transformation import TransformationBase
from csdl_alpha.src.transformations.transformation import TransformationBase
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import get_type_string, variablize

from typing import Callable 

class DataAssimilation(TransformationBase):

    def post_init(self):
        self.surrogate_injections = []

    def add_correction(
            self,
            sources:list[Variable],
            targets:list[Variable],
            surrogate_model:Callable,
            target_combination:str = None,
        ):
        # Check types
        if isinstance(sources, Variable):
            sources = [sources]
        else:
            for source in sources:
                if not isinstance(source, Variable):
                    raise ValueError(f"Arg \'sources\' expected type \'Variable\' or \'list[Variable]\', got {get_type_string(source)}")
        if isinstance(targets, Variable):
            targets = [targets]
        else:
            for target in targets:
                if not isinstance(target, Variable):
                    raise ValueError(f"Arg \'targets\' expected type \'Variable\' or \'list[Variable]\', got {get_type_string(target)}")
        if not callable(surrogate_model):
            raise TypeError(f"Arg \'surrogate_model\' expected type \'Callable\', got {get_type_string(surrogate_model)}")

        self.surrogate_injections.append({
            'sources': sources,
            'targets': targets,
            'surrogate_model': surrogate_model,
            'target_combination': target_combination
        })

    def inject_to_graph(
            self,
            sources:list[Variable],
            targets:list[Variable],
            surrogate_model:Callable,
            target_combination:str = None,
            index:int = 0,
        ):
        # Get current recorder
        recorder = self.recorder
        active_graph = recorder.active_graph
        thetas = {}

        # For all outputs, disconnect it from the graph and replace it with a dummy output for pooling
        for target in targets:
            preceding_operation = active_graph.predecessors(target)[0]
            dummy_output = Variable(shape = target.shape, value = target.value)
            preceding_operation.replace_output_variable(target, dummy_output)
    
        # Now evaluate the surrogate model on the sources
        with csdl.namespace(f'{index}'):
            # Preprocess inputs and apply correction function to all inputs
            # The surrogate model takes in a single, flattened input
            if len(sources) == 1:
                if sources[0] is None:
                    sources[0] = dummy_output.flatten()
                corrected_variable, parameters = surrogate_model(sources[0].flatten())
            else:
                flattened_inputs = []
                for input_var in sources:
                    if input_var is None:
                        flattened_inputs.append(dummy_output.flatten())
                    else:
                        flattened_inputs.append(input_var.flatten())
                    
                concatenated_inputs = csdl.concatenate(flattened_inputs)
                corrected_variable, parameters = surrogate_model(concatenated_inputs.flatten())

        # Store design variables
        for index_inner, param_info in enumerate(parameters):
            param_var = param_info['var']
            name = f'{index}_{index_inner}_{param_var.name}'
            thetas[name] = {
                'var': param_var,
                'value': param_info['value']
            }

    def apply(self):
        self.recorder = self.get_current_recorder()
        all_thetas:dict = {}
        i:int = 0
        for injection in self.surrogate_injections:
            sources = injection['sources']
            targets = injection['targets']
            surrogate_model = injection['surrogate_model']
            target_combination = injection['target_combination']

            # Transform graph
            thetas = self.inject_to_graph(
                sources,
                targets,
                surrogate_model,
                target_combination,
                i,
            )
            i += 1
