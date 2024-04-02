

class GraphError(ValueError):
    def __init__(
            self,
            message: str,
            tag: str = None,
            relevant_nodes = None):
        '''Unable to perform graph transformation'''
        self.message = message
        self.tag = tag
        self.relevant_nodes = relevant_nodes
        super().__init__(self.message)