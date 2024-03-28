from csdl_alpha.utils.parameters import Parameters

class Model:
    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Model class.

        Args:
            **kwargs: Additional keyword arguments to be passed to the Parameters class.

        """
        self.parameters = Parameters()
        self.evaluate = self._apply_namespacing(self.evaluate)
        self.parameters.hold(kwargs)
        self.initialize()
        self.parameters.check(kwargs)

    def _apply_namespacing(self, evaluate): # TODO: seperate this from model
        """
        Applies namespacing to the evaluate method.

        Args:
            evaluate: The evaluate method to be namespaced.

        Returns:
            The namespaced evaluate method.

        """
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
        """
        user-defined method to initialize the model.

        Args:
            *kwargs: Additional keyword arguments.

        """
        pass

    def evaluate(self):
        """
        User-defined method to evaluate the model.
        Name-spacing is applied to this method via 'name' kwarg.

        """
        pass
