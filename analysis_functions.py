from joblib import hash

import functools
from tqdm import tqdm
import warnings




class CoProcessor:

    def __init__(self, cos):
        self.cos = cos # to be edited
        self.coskey = cos # save very first input
        self.mem = {}
        self.call_count = 0  # Counter to keep track of function calls



    def _data_structure_checks(self, orig_val, new_val, return_new_val=False):
        # 1) enforce same “class” (so subclasses are allowed too)
        if not isinstance(new_val, orig_val.__class__):
            warnings.warn(
                f"All objects are not of the same type. Can't use self.interact method.",
                UserWarning,
                stacklevel=2
            )
    
        # 2) if the original has a `.shape`, require same shape
        orig_shape = getattr(orig_val, "shape", None)
        if orig_shape is not None:
            new_shape = getattr(new_val, "shape", None)
            if new_shape is None:
                warnings.warn(
                    f"Expected shaped object, got {type(new_val).__name__} for one of the objects. Can't use self.interact method.",
                    UserWarning,
                    stacklevel=2
                )
            elif new_shape != orig_shape:
                warnings.warn(
                    f"Shape mismatch: {new_shape} != {orig_shape}. Can't use self.interact method.",
                    UserWarning,
                    stacklevel=2
                )
        if return_new_val:
            return new_val
       

    def _count_calls(self, func): # This function is not working as expected so ignore...
        """
        Private helper that wraps the splitter/transformer/interactor function to count how many times it is called.
        
        Parameters
        ----------
        func : function
            The original splitter/transformer/interactor function.
            
        Returns
        -------
        function
            The wrapped splitter/transformer/interactor function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.call_count += 1
            return func(*args, **kwargs)
        return wrapper


    def split(self, splitter, update=True):

        """
        Function applies transformer function on the individual comonomers, while using memoisation to avoid recomputation of identical comonomer units
        
        Inputs
        ------
        
        splitter : function(x)
            - Takes single x and splits into parts where each part becomes an item in tuple
            - Uses memoisation to avoid recomputation
        
        update : Bool
            option to save the output to self.cos, overwriting previous
        
        Returns
        -------
        List 
            A List of Tuples containing the splitted items
    
        Example usage
        --------------
        items = ['PEEK-PEDEK', 'PES-PEEK']
        cp = CoProcessor(items)
        cp.split(lambda x: x.split('-'))
        Returns -> [('PEEK', 'PEDEK'), ('PES', 'PEEK')]


        
        """
        splitter = self._count_calls(splitter)
        cos=self.cos
        sp_cos = []
        mem = {}  # Dictionary for memoization
    
        if isinstance(cos, list):  # If cos is a list
            for co in cos:  # Loop through copolymers
                key = hash(co)                   
                if key in mem:  # If already computed, reuse it
                    sp_co = mem[key]
                    # print("Memoised")
                else:
                    sp_co = splitter(co)  # Compute and store in mem
                    mem[key] = sp_co
                    # print("Computed")

                sp_cos.append(tuple(sp_co))  # Convert list to tuple before storing

        else:  # If cos is a single tuple
            sp_cos = tuple(splitter(cos))

        if update:
            self.cos = sp_cos
        
        self.mem = mem    
        return sp_cos
    
        
    def transform(self, transformer, update=True, progress_bar=False):
        """
        Applies transformer function to individual comonomers with optional progress bar and memoization.

        Parameters
        ----------
        transformer : function(x)
            Function to apply to each comonomer unit.
        update : bool, optional
            If True, updates self.cos with the transformed values.
        progress_bar : bool, optional
            Whether to display a progress bar.

        Returns
        -------
        list or tuple
            Transformed comonomers.

        Example
        -------
        items = [('PEK','PEEK'), ('PES','PEEK')]
        cp = CoProcessor(items)
        cp.transform(lambda x: x)
        """
        transformer = self._count_calls(transformer)
        cos = self.cos
        tr_cos = []
        mem = {}  # Dictionary for memoization
        # orig_co_i = flatten(cos, flatten_tuples=True)[0]
        tr_co_i_previous = None
        if isinstance(cos, list):  # List of tuples
            iterable = tqdm(cos, desc="Transforming copolymers") if progress_bar else cos
            for co in iterable:
                tr_co = []
                for co_i in co:
                    key = hash(co_i)
                    if key in mem:
                        tr_co_i = mem[key]
                    else:
                        # check the type of co_i
                        orig_co_i = co_i
                        tr_co_i = transformer(co_i)
                        if tr_co_i_previous is None:
                            pass
                        else:
                            self._data_structure_checks(tr_co_i_previous, tr_co_i)
                        tr_co_i_previous = tr_co_i # I want to make a copy of the previous transformed iterable to conduct the data structure checks below
                        mem[key] = tr_co_i
                    tr_co.append(tr_co_i)
                tr_cos.append(tuple(tr_co))
        else:  # Single tuple
            tr_cos = tuple([transformer(co_i) for co_i in cos])
    
        if update:
            self.cos = tr_cos

        self.mem = mem
        return tr_cos

        
        
    def interact(self, interactor, symmetry='permutation', update=True, progress_bar=False):
        """
        Applies an interactor function to comonomers with optional symmetry handling and progress bar.

        Parameters
        ----------
        interactor : function(tuple)
            Function to compute interactions from a comonomer tuple.
        symmetry : {'permutation', 'mirror', 'none'}, optional
            Symmetry mode to reduce redundant computation.
        update : bool, optional
            Whether to update self.cos with interaction results.
        progress_bar : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        list
            Interaction results.

        Example 1
        ---------
        cosmiles = [('CC', 'C'), ('C', 'CC'), ('C', 'CC'), ('N', 'C(=O)'), ('c1c(cccc1)', 'C')]
        cp = CoProcessor(cosmiles)
        transformer = lambda x : rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(x))
        cp = CoProcessor(cosmiles)
        cp.transform(transformer)
        interactor = lambda co : np.divide(*co) # interactor((co[0] , co[1])) != interactor(co[1] , co[0])
        cp.interact(interactor=interactor, symmetry='none')

        Example 2
        ---------
        conum = [(np.array([1,2,3]), np.array([3,2,1])),
         (np.array([1.3,2.2,3.1]), np.array([31,22,11])),
         (np.array([11,12,13]), np.array([23,22,21]))
        ]
        cp = CoProcessor(conum)
        interactor = lambda co : np.sum(co)
        cp.interact(interactor, symmetry='permutation')

        Example 3
        ---------
        coseq = [('E', 'C', 'A'), ('A', 'E', 'C'), ('W', 'A'), ('A', 'W'), ('K', 'J', 'E', 'M'), ('M', 'E', 'J', 'K')]
        cp = CoProcessor(coseq)
        # cp.transform(transformer)
        interactor = lambda co : ''.join(co)
        cp.interact(interactor=interactor, symmetry='mirror')
        """
        interactor = self._count_calls(interactor)
        mem = {}
        in_cos = []

        iterable = tqdm(self.cos, desc="Computing interactions") if progress_bar else self.cos
        for co in iterable:
            key = [hash(k) for k in co]
            if symmetry == 'permutation':
                key = tuple(sorted(key))
            elif symmetry == 'mirror':
                key = tuple(min(key, key[::-1]))
            elif symmetry == 'none':
                key = tuple(key)
            else:
                raise ValueError("Select a valid symmetry value from: 'permutation', 'mirror', 'none'.")
            
            if key in mem:
                in_co = mem[key]
            else:
                in_co = interactor(co)
                mem[key] = in_co
            in_cos.append(in_co)
        
        if update:
            self.cos = in_cos
        self.mem = mem
        return in_cos

    def get_cosmap(self):
        """
        Creates a dictionary mapping the original inputs to the current state.

        Returns
        -------
        dict
        Example
        -------
        cosmiles = [('CC', 'C'), ('C', 'CC'), ('C', 'CC'), ('N', 'C(=O)'), ('c1c(cccc1)', 'C')]
        cp = CoProcessor(cosmiles)
        transformer = lambda x : rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(x))
        cp.transform(transformer)
        interactor = lambda co : np.sum(co)
        cp.interact(interactor, symmetrical=True)
        cp.get_cosmap()
        """
        return {k: v for k, v in zip(self.coskey, self.cos)}

    

    def edit_cos(self, mapto: dict):
        """
        Edit one (or a subset of) self.cos value(s) by a specified self.coskey item
        eg. if self.coskey -> [('PEDEPK', 'PEPEPK'),('PEPKPK', 'PEPEPKPK'), ('PEPEPKPK', 'PEPEPK')] 
        and self.cos ->  [(array([ 7.73,  5.97 , 38.9]), array([ 7.73,  5.97 , 10.3])),
                          (array([ 7.73,  5.97 , 10.3]), array([ 7.73,  5.97 , 10.3])),
                          (array([ 7.73,  5.97 , 10.3]), array([ 7.73,  5.97 , 10.3]))]
                          
        noting that coskey and cos are quite general in their sizes e.g. could be a list of 3 itemed tuples
        but generally have the structure of either a list of tuples or a single tuple.

        Parameters
        ----------
        mapto : dict
            mapping from a key in one of the self.coskey tuples
            to the new value you want in self.cos.
        update : bool
            passed through to self.transform
        progress_bar : bool
            passed through to self.transform

        Example
        -------
        items = [(1,'PEEK'), ('PES','PEEK')]
        cp = CoProcessor(items)
        cp.edit_cos(mapto={1:'PEK'})
        """
        

        for wantkey in mapto.keys():
            if wantkey not in flatten(list(self.coskey), flatten_tuples=True):
                raise KeyError(f"Key '{wantkey}' not found in any self.coskey")

        def convert_tuples(keys, values):
            return tuple(mapto[key] if key in mapto else val for key, val in zip(keys, values))
        
        if isinstance(self.cos, list):
            self.cos = [convert_tuples(keys, values) for keys, values in zip(self.coskey, self.cos)]

        else:
            self.cos = convert_tuples(self.coskey, self.cos)
        
        return self.transform(lambda x : x) # run through for appropriate data checks

    

    def zip(self, *items, update=True):
        """
        Zips self.cos with all the provided iterable items. All iterables (self.cos and items)
        must have the same length.
    
        Parameters:
            *items (iterables): One or more iterables to be zipped with self.cos.
            update (bool): If True, update self.cos with the resulting zipped list.
    
        Returns:
            list of tuples: The zipped result containing tuples of corresponding elements.
    
        Raises:
            ValueError: If any of the provided iterables do not have the same length as self.cos.
        """
        # Check that each item in items has the same length as self.cos.
        for idx, item in enumerate(items, start=1):
            if len(item) != len(self.cos):
                raise ValueError(
                    f"Length mismatch: self.cos has length {len(self.cos)}, "
                    f"but item {idx} has length {len(item)}"
                )
        
        # Zip self.cos with all other items.
        newcos = list(zip(self.cos, *items))
        
        # Optionally update self.cos.
        if update:
            self.cos = newcos
            
        return newcos





def flatten(data, flatten_tuples=False):
    """
    Recursively flatten a nested list—and optionally tuples.

    Args:
        data (list or tuple): The nested structure.
        flatten_tuples (bool): If True, tuples are flattened just like lists.
                                If False, tuples are returned as atomic elements.

    Returns:
        list: A flattened version of the input.
    """
    # Check for list, or tuple only when flatten_tuples is True
    is_container = isinstance(data, list) or (flatten_tuples and isinstance(data, tuple))
    if is_container:
        return [item for sublist in data for item in flatten(sublist, flatten_tuples)]
    else:
        return [data]


