from csdl_alpha.src.recorder import Recorder

class RecManager(object):
    instantiated = False
    def __init__(self) -> None:
        """
        Manager class acts like the "global state".
        ONLY ONE INSTANCE ALLOWED
        """
        if RecManager.instantiated:
            raise Exception("DEV ERROR: RecManager already instantiated. Only one instance should ever be created")

        self.active_recorder: Recorder = None
        self.constructed_recorders: list[Recorder] = []
        self.recorder_stack: list[Recorder] = []

        RecManager.instantiated = True

    def __repr__(self) -> str:
        active_recorder = self.active_recorder
        output_string = f"\nActive Recorder: {active_recorder}"
        
        # print all constructed recorders in order
        for i, recorder in enumerate(self.constructed_recorders):
            output_string += f"\nRecorder {i}: {recorder}"

            if recorder == active_recorder:
                output_string += " (active)"
        output_string += "\n"
    
        return output_string
    
    def activate_recorder(self, recorder: Recorder):
        self.recorder_stack.append(recorder)
        self.update_active_recorder()

    def deactivate_recorder(self, recorder: Recorder):
        if self.active_recorder != recorder:
            raise Exception("DEV ERROR: Deactivating a recorder that is not active")
        self.recorder_stack.pop()
        self.update_active_recorder()

    def update_active_recorder(self):
        if len(self.recorder_stack) > 0:
            self.active_recorder = self.recorder_stack[-1]
        else:
            self.active_recorder = None

# manager = RecManager()

# def get_current_recorder():
#     return manager.active_recorder

# def build_new_recorder():
#     return Recorder()

# def print_all_recorders():
#     print(manager)