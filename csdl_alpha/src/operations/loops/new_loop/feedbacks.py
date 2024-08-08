from csdl_alpha.src.graph.variable import Variable

class Feedback(object):
    def __init__(self):
        self.external_input:Variable = None
        self.internal_input:Variable = None
        self.output:Variable = None

        self.fully_defined:bool = False
        self.shape:tuple[int] = None

    def process_shape(self, var:Variable):
        if self.shape is None:
            self.shape = var.shape
        else:
            if self.shape != var.shape:
                raise ValueError(f"Shape mismatch for feedback var {var.info()}. expected: {self.shape}, got: {var.shape}")

    def set_external_input(self, external_input:Variable):
        if self.external_input is not None:
            raise ValueError("External input is already set.")
        self.external_input = external_input
        self.process_shape(external_input)

        if self.internal_input is not None and self.output is not None:
            self.fully_defined = True

    def set_internal_input(self, internal_input:Variable):
        if self.internal_input is not None:
            raise ValueError("Internal input is already set.")
        self.internal_input = internal_input
        self.process_shape(internal_input)

        if self.external_input is not None and self.output is not None:
            self.fully_defined = True

    def set_output(self, output:Variable):
        if self.output is not None:
            raise ValueError("Internal output is already set.")
        self.output = output
        self.process_shape(output)

        if self.external_input is not None and self.internal_input is not None:
            self.fully_defined = True

    def set_triple(self, external_input:Variable, internal_input:Variable, output:Variable):
        self.set_external_input(external_input)
        self.set_internal_input(internal_input)
        self.set_output(output)

class Feedbacks(object):
    def __init__(self):
        self._int_input_to_feedback:dict[Variable, Feedback] = {}
        self._output_to_feedback:dict[Variable, Feedback] = {}
        self._external_in_to_feedback:dict[Variable, Feedback] = {}

    def initialize_feedback(self, ext_input_var:Variable)->Variable:
        internal_input_var = Variable(
            shape = ext_input_var.shape,
            value = ext_input_var.value,
        )
        if internal_input_var in self._int_input_to_feedback:
            raise ValueError(f"Internal input {internal_input_var.info()} already initialized.")
        self._int_input_to_feedback[internal_input_var] = Feedback()
        self._int_input_to_feedback[internal_input_var].set_external_input(ext_input_var)
        self._int_input_to_feedback[internal_input_var].set_internal_input(internal_input_var)

        self._external_in_to_feedback[ext_input_var] = self._int_input_to_feedback[internal_input_var]
        return internal_input_var

    def finalize_feedback(
            self,
            int_input_var:Variable,
            output:Variable,
            )->None:
        if int_input_var not in self._int_input_to_feedback:
            raise ValueError(f"Feedback of variable {int_input_var.info()} not initialized.")
        feedback = self._int_input_to_feedback[int_input_var]
        feedback.set_output(output)

        # update mapping from output to feedback
        self._output_to_feedback[output] = feedback
        return output
    
    def set_triple(
            self,
            ext_input_var:Variable,
            int_input_var:Variable,
            output:Variable,
        )->None:
        if int_input_var in self._int_input_to_feedback:
            raise ValueError(f"Feedback for internal input {int_input_var.info()} already exists.")
        if output in self._output_to_feedback:
            raise ValueError(f"Feedback for output {output.info()} already exists.")
        feedback = Feedback()
        feedback.set_triple(ext_input_var, int_input_var, output)
        self._int_input_to_feedback[int_input_var] = feedback
        self._output_to_feedback[output] = feedback
        self._external_in_to_feedback[ext_input_var] = feedback

    def check(self):
        for int_input, feedback in self._int_input_to_feedback.items():
            if not feedback.fully_defined:
                raise ValueError("Feedbacks are not fully defined.")
            if feedback.output not in self._output_to_feedback:
                raise ValueError("INTERNAL ERROR: Output mapping incorrect in feedbacks.")
            
        if len(self._int_input_to_feedback) != len(self._output_to_feedback):
            raise ValueError("INTERNAL ERROR: Output/input feedback mapping incorrect.")

    def __repr__(self) -> str:
        string = ""
        for int_input, feedback in self._int_input_to_feedback.items():
            string += f"Feedback: {feedback.shape}\n"
            string += f"\tInternal input: {int_input}\n"
            string += f"\tExternal input: {feedback.external_input}\n"
            string += f"\tOutput: {feedback.output}\n"
        return string