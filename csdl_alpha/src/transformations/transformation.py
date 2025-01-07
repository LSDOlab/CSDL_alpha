
from typing import Union

class TransformationBase(object):
    
    def __init__(self, name = None, print_info:bool = False):
        self.graph_actions:dict[str, dict] = {
            'add': {},
            'del': {},
            'transform': {},
        }
        self.name:str = name
        self._status:int = 0

        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError(build_type_error_string('name', str, self.name))
        else:
            self.name = 'default name'

        self.post_init()

    def post_init(self):
        pass

    def change_to_active_status(self):
        self._status = 1

    def change_to_locked_status(self):
        self._status = 2

    def get_current_recorder(self):
        import csdl_alpha as csdl
        recorder = csdl.get_current_recorder()
        return recorder

    def apply(self):
        raise NotImplementedError(f'Transformation {self} not implemented')

    def record_action(self,action_type:str, actions:Union[list, 'TransformationBase'])->None:

        # Can't record actions if inactive transform as transforms are immutable.
        if self._status == 0:
            raise ValueError('Transformation is not yet active')
        elif self._status == 2:
            raise ValueError('Transformation has already been deactivated')

        if not isinstance(actions, list):
            actions = [actions]
        if action_type not in self.graph_actions:
            raise KeyError(f'action_type {action_type} does not exist')
        for action in actions:
            self.graph_actions[action_type][action] = 0

    def transform(self, *args, **kwargs)->None:
        import csdl_alpha as csdl
        
        # automatically find current recorder
        active_recorder:csdl.Recorder = csdl.get_current_recorder()

        # pre-processing
        active_recorder.transformation_logger.push(self)

        # apply transfomration here:
        self.apply(*args, **kwargs) # change to decorator to keep linting?
    
        # post-processing
        active_recorder.transformation_logger.pop()

    def info(self)->str:
        string = f'Transformation {self.name} info:'
        for key in self.graph_actions:
            num_actions = len(self.graph_actions[key])
            string+= f'\n\t{key}: {num_actions} actions'
        return string

class UserTransformation(TransformationBase):
    def __init__(self):
        super().__init__(name='UserTransform')
        self.change_to_active_status()

class TransformationLogger():

    def __init__(self):
        base_transformation = UserTransformation()
        self.call_history:list[TransformationBase] = [base_transformation]
        self.call_tree:dict[TransformationBase,dict] = {base_transformation: set()}
        self.call_stack:list[TransformationBase] = [base_transformation]

    def push(self, t:TransformationBase):
        if not isinstance(t, TransformationBase):
            raise TypeError(build_type_error_string('t', TransformationBase, t))
        current_transformation = self.get_current()

        # update state
        self.call_history.append(t)
        self.call_stack.append(t)
        self.call_tree[current_transformation].add(t)
        self.call_tree[t] = set()

        t.change_to_active_status()
    
    def pop(self):
        finished_transform = self.call_stack.pop()
        finished_transform.change_to_locked_status()
        return self.get_current()
    
    def get_current(self):
        return self.call_stack[-1]
    
    def _to_str(self):
        return 
    
    def visualize(self):
        raise NotImplementedError('visualization not yet implemented')
    
    def get_history_string(self):
        hist_str = f'Transformation Log {self}: Call History:'
        hist_str += transformation_list_to_str(self.call_history)
        return hist_str
    
    def get_stack_string(self):
        stack_str = f'Transformation Log {self}: Call Stack:'
        stack_str += transformation_list_to_str(self.call_stack)
        return stack_str

    def print_history(self):
        return print(self.get_history_string())
    
    def print_tree(self):
        return
    
    def print_stack(self):
        return print(self.get_stack_string())
    
def transformation_list_to_str(
        transformation_list:list[TransformationBase],
        show_status = True,
        ):
    string = ''
    for i, transformation in enumerate(transformation_list):
        string += f'\n{i})\t{transformation.name}'
        if show_status:
            string += f' (status: {transformation._status})'

    return string

def build_type_error_string(
        arg_name:str,
        arg_expected_type,
        arg_obj,
    )->str:
    from csdl_alpha.utils.inputs import get_type_string
    return f'Argument \'{arg_name}\' expected type \'{arg_expected_type.__name__}\', {get_type_string(arg_obj)} given.'