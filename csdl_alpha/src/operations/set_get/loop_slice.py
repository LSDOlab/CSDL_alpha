from csdl_alpha.src.operations.set_get.slice import Slicer, Slice

class LoopSlicer(Slicer):
    def __init__(self):
        from csdl_alpha.src.graph.variable import Variable
        super().__init__()
        self.valid_types = self.valid_types + (Variable,)

    def __getitem__(self, keys):
        from csdl_alpha.src.graph.variable import Variable
        from csdl_alpha.api import get_current_recorder

        # tuplify keys if not already a tuple
        if isinstance(keys, self.valid_types):
            keys = (keys,)
        elif not isinstance(keys, tuple):
            raise TypeError(f"Invalid key {keys}. Must be an integer, slice, or tuple index list.")
        
        # check to make sure all axes are correct types and dimensions
        return_keys = []
        mapping_list = []

        var2maplist_index = {}
        current_list_length = None
        for i, k in enumerate(keys):
            # check to make sure types are correct
            current_list_length = self.check_axis_validity(k, return_keys, current_list_length)

            # We need to map CSDL variables to slice template.
            if isinstance(k, Variable): #integer
                self.add_to_mapping_list(k, mapping_list, var2maplist_index, i)
            elif isinstance(k, slice): # slice
                if isinstance(k.start, Variable) and isinstance(k.stop, Variable):
                    # Need to parse the graph between these to variables to see if their difference is constant
                    graph = get_current_recorder().active_graph
                    nodes_between = graph._get_intersection([k.start], [k.stop], check_sources=False, check_targets=False)
                    if len(nodes_between) == 0:
                        nodes_between = graph._get_intersection([k.stop], [k.start], check_sources=False, check_targets=False)
                    




                if isinstance(k.start, Variable):
                    raise TypeError("Slice start cannot be a CSDL variable")
                    self.add_to_mapping_list(k.start, mapping_list, var2maplist_index, (i, 's'))
                if isinstance(k.stop, Variable):
                    raise TypeError("Slice stop cannot be a CSDL variable")
                    self.add_to_mapping_list(k.stop, mapping_list, var2maplist_index, (i, 'e'))
                if isinstance(k.step, Variable):
                    raise TypeError("Slice step cannot be a CSDL variable")
            elif isinstance(k, list): # list
                if len(k) == 1:
                    if isinstance(k[0], Variable):
                        self.add_to_mapping_list(k[0], mapping_list, var2maplist_index, i)
                else:
                    for j, l in enumerate(k):
                        if isinstance(l, Variable):
                            self.add_to_mapping_list(l, mapping_list, var2maplist_index, (i, j))

        return VarSlice((return_keys), tuple(mapping_list))

    def add_to_mapping_list(
            self,
            k,
            mapping_list,
            var2maplist_index,
            index_mapping_value,
        ):

        if k not in var2maplist_index:
            var2maplist_index[k] = len(mapping_list)
            mapping_list.append((k, []))

        mapping_list[var2maplist_index[k]][1].append(index_mapping_value)


_loop_slice = LoopSlicer()


class VarSlice(Slice):

    def __init__(self, keys:list, mapping_list:tuple) -> None:
        """
        -slice_template: (1,[3,i], j:j+1, i)

        evaluate method maps from arguments (i,j, j+1) to template locations:
        
        var2slice_map: ([1,1], [3]), ([2, start]), ([2,end])
        """

        self.slices = keys
        self.vars = []
        self.var2slicemap = []

        from csdl_alpha.src.graph.variable import Variable
        for arg_index, (var, maps) in enumerate(mapping_list):
            # arg_index is the index of the CSDL variable
            self.vars.append(var)

            # Integer if list of integer
            current_arg_map = []
            for map in maps:
                if isinstance(map, int): # integer
                    current_arg_map.append((self.map2int, (map)))
                else:
                    if isinstance(map[1], str):
                        current_arg_map.append((self.map2slice, map))
                    elif isinstance(map[1], int):
                        current_arg_map.append((self.map2list, map))
            
            self.var2slicemap.append(tuple(current_arg_map))

        self.vars = tuple(self.vars)

    def map2int(self, map:int, arg_int):
        """
        replaces an integer in the slice at index map with arg_int
        """
        self.slices[map] = arg_int

    def map2list(self, map:tuple[int,int], arg_int):
        self.slices[map[0]][map[1]] = arg_int

    def map2slice(self, map:tuple[int,str], arg_int):
        cur_slice = self.slices[map[0]]
        if map[1] == 's':
            self.slices[map[0]] = slice(arg_int, cur_slice.stop, cur_slice.step)
        elif map[1] == 'e':
            self.slices[map[0]] = slice(cur_slice.start, arg_int, cur_slice.step)


    def evaluate_zeros(self):
        """
        Returns the shape of the slice
        """
        return self.evaluate(*[0 for _ in self.vars])

    def evaluate(self, *args:tuple[float])->tuple:

        for arg_index, arg_value in enumerate(args):
            # arg_index is the index of the CSDL variable
            # arg_value is the value of that CSDL variable

            arg_int = int(arg_value) # value that has been cast to an integer to replace slice variable

            maps = self.var2slicemap[arg_index]
            for map in maps:
                map[0](map[1], arg_int)
        return tuple(self.slices)

