from .src.data import inline_save, import_h5py, save_optimization_variables, save_all_variables
from .src.graph.variable import Variable
from .src.model import Model
from .manager import RecManager
manager = RecManager()

def get_current_recorder():
    return manager.active_recorder

def build_new_recorder(inline = False, debug = False, expand_ops = False, auto_hierarchy = False):
    from .src.recorder import Recorder
    return Recorder(inline = inline, debug=debug, expand_ops=expand_ops, auto_hierarchy=auto_hierarchy)

def enter_namespace(namespace: str):
    """
    Enters a new namespace.

    Args:
        namespace: The name of the namespace to enter.
    """
    recorder = get_current_recorder()
    recorder._enter_namespace(namespace)

def exit_namespace():
    """
    Exits the current namespace.
    """
    recorder = get_current_recorder()
    recorder._exit_namespace()


def enter_subgraph():
    """
    Enters a new subgraph.
    """
    recorder = get_current_recorder()
    recorder._enter_subgraph()

def exit_subgraph():
    """
    Exits the current subgraph.
    """
    recorder = get_current_recorder()
    recorder._exit_subgraph()

def print_all_recorders():
    print(manager)
    