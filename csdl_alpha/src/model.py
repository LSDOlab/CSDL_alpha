from csdl_alpha.utils.parameters import Parameters

class Model:
    def __init__(self, **kwargs):
        self.parameters = Parameters()
        self.evaluate = self._apply_namespacing(self.evaluate)
        self.parameters.hold(kwargs)
        self.initialize()
        self.parameters.check(kwargs)

    def _apply_namespacing(self, evaluate):
        def new_evaluate(*args, **kwargs):
            from csdl_alpha.api import enter_namespace, exit_namespace

            if 'name' in kwargs:
                name = kwargs['name']
                del kwargs['name']
            else:
                name = self.__class__.__name__
            enter_namespace(name)
            outputs = evaluate(*args, **kwargs)
            exit_namespace()
            return outputs
        return new_evaluate
    
    def initialize(self, *kwargs):
        pass

    def evaluate(self):
        pass

