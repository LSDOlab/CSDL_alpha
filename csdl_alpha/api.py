from .src.graph.variable import Variable, ImplicitVariable
from .src.model import Model
from .manager import RecManager
manager = RecManager()

def get_current_recorder():
    return manager.active_recorder

def build_new_recorder(inline = False):
    from .src.recorder import Recorder
    return Recorder(inline = inline)

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
    