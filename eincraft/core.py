import numpy as np
from typing import Literal

class EinTenAddend:

    def __init__(self, name:str) -> None:
        self.name = name

        self.prefactor = 1.0
        self.indices_and_tensors = []

        # for the einsum last index, otherwise it is implicit
        self.index = None

        # to cache the einsum_path
        self.einsum_path = None

    def reorder(self, indices, max_index):
        '''Reorder the tensor according to the numeric indices, 
           The tensor is assume to be constructed with ordered indices (0, 1, 2, ...)
           If used as an intermediate tensor, the indices should be ordered as requested
           Need max_index to keep track of the highest index assigned
           for the dummy indices needed by the tensor
        '''
        if len(self.indices_and_tensors) == 0:
            return [(indices, self)], max_index

        mapping = {index: i for index, i in enumerate(indices)}

        indices_and_tensors = []

        for indices, tensor in self.indices_and_tensors:
            numeric_indices = []
            for index in indices:
                if index not in mapping:
                    mapping[index] = max_index + 1
                    max_index = mapping[index]
                numeric_indices.append(mapping[index])
            indices_and_tensors.append((numeric_indices, tensor))

        return indices_and_tensors, max_index

    def get_subscripted(self, indices):
        return EinTenAddendSubscripted([(list(indices), self)], prefactor=self.prefactor)

    def update_from_indexed(self, indices, einten_indexed):
        indices_and_tensors = einten_indexed.convert_indices_and_tensors(list(indices))
        self.prefactor = einten_indexed.prefactor
        self.indices_and_tensors = indices_and_tensors
        self.index = [i for i in range(len(indices))]
        self.einsum_path = None

    def with_fixed(self, tensor, mode: Literal['identity', 'diagonal']= 'identity'):
        ''' 
        Set the tensor to the identity to semplify the operations
        '''
        if mode not in ['identity', 'diagonal']:
            raise ValueError(f"Mode {mode} is not supported to fix tensors")

        result = EinTenAddend(f'{self.name}_{tensor.name}_identity')
        result.prefactor = self.prefactor
        result.index = self.index

        new_indices_and_tensors = []

        indices_to_set_equal = {}
        for indices, ten in self.indices_and_tensors:
            if ten.name == tensor.name:
                target_index = None
                for index in indices:
                    if index in result.index:
                        if target_index is not None:
                            raise ValueError(f"Index {index} is repeated in tensor {tensor.name}, einsum does not allow repeated indices in the result")
                        target_index = index
                if target_index is None:
                    target_index = indices[0]

                for index in indices:
                    if index in result.index and index != target_index:
                        raise ValueError(f"Index {index} is repeated in tensor {tensor.name}, einsum does not allow repeated indices in the result")
                    if index in indices_to_set_equal and indices_to_set_equal[index] != target_index:
                        raise ValueError(f"Index {index} is repeated in tensor {tensor.name}, einsum does not allow repeated indices in the result")
                    indices_to_set_equal[index] = target_index

                if mode == 'diagonal':
                    new_indices_and_tensors.append(([target_index], ten))
            else:
                new_indices_and_tensors.append((indices, ten))

        for indices, ten in new_indices_and_tensors:
            for i, index in enumerate(indices):
                if index in indices_to_set_equal:
                    indices[i] = indices_to_set_equal[index]

        result.indices_and_tensors = new_indices_and_tensors

        return result

    def evaluate(self, **kwargs):
        '''Apply the tensor to the arguments
        Construct the einsum arguments and calling np.einsum
        '''

        # Construct the einsum arguments
        args = [(kwargs[ten.name], indices) for indices, ten in self.indices_and_tensors]
        # Flatten the list
        args = [arg for pair in args for arg in pair]
        args.append(self.index)

        #print([(ten.name, indices) for indices, ten in self.indices_and_tensors], self.index, self.prefactor)

        if self.einsum_path is None:
            self.einsum_path, _ = np.einsum_path(*args, optimize=True) 
           #for line in _.split('\n'):
           #    if 'Optimized scaling:  ' in line:
           #        print(line)
           #        break

        return self.prefactor * np.einsum(*args, optimize=self.einsum_path)

    def print(self):
        return f"{self.prefactor} * " + ' * '.join([f"{ten.name}_{''.join([chr(97 + i) for i in indices])}" for indices, ten in self.indices_and_tensors]) 

    def evaluate_screened(self, **kwargs):
        '''Apply the tensor to the arguments
        Construct the einsum arguments and calling np.einsum
        '''

        ndim = len(self.index)

        shape = [None] * ndim

        for indices, ten in self.indices_and_tensors:
            for i, index in enumerate(indices):
                access_arg = []
                if index < ndim:
                    if shape[index] is None:
                        shape[index] = kwargs[ten.name].shape[i]
                    elif shape[index] != kwargs[ten.name].shape[i]:
                        raise ValueError(f"Dimension mismatch for index {index} in tensor {ten.name}")
                else:
                    access_arg.append(None)

        result = np.zeros(shape)

        args = [(kwargs[ten.name], indices) for indices, ten in self.indices_and_tensors]
        args = [arg for pair in args for arg in pair]
        args.append([])

        for element in np.ndindex(*shape):
            result[element] = self.prefactor * np.einsum(*args, optimize=True)

        return result


        # Construct the einsum arguments
        args = [(kwargs[ten.name], indices) for indices, ten in self.indices_and_tensors]

        # Flatten the list
        args = [arg for pair in args for arg in pair]
        if self.index:
            args.append(self.index)

        #print([(ten.name, indices) for indices, ten in self.indices_and_tensors], self.index, self.prefactor)

        if self.einsum_path is None:
            self.einsum_path, _ = np.einsum_path(*args, optimize=True) 
           #for line in _.split('\n'):
           #    if 'Optimized scaling:  ' in line:
           #        print(line)
           #        break

        return self.prefactor * np.einsum(*args, optimize=self.einsum_path)

class EinTenAddendSubscripted:

    def __init__(self, indices_and_tensors: list, prefactor=1.0) -> None:
        self.indices_and_tensors = indices_and_tensors
        self.prefactor = prefactor

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            result = EinTenAddendSubscripted(self.indices_and_tensors, prefactor=self.prefactor * other)
        elif isinstance(other, EinTenAddendSubscripted):
            result = EinTenAddendSubscripted(self.indices_and_tensors + other.indices_and_tensors,
                                   prefactor=self.prefactor * other.prefactor)
        else:
            raise ValueError(f"Multiplication with type {type(other)} is not supported.")

        return result

    def convert_indices_and_tensors(self, indices):
        '''Convert the operations indices to numbers
        '''
        
        '''
        Map the indices to numbers
        '''

        mapping = {}

        for letter in indices:
            if letter not in mapping:
                mapping[letter] = len(mapping)

        '''
        TODO: einsum does not allow repeated indices
        but we could allow it if are repeated only in the intermediate tensors
        '''
        #if len(mapping) != len(indices):
        #    raise ValueError(f"Indices {indices} are not unique in the output")

        # check max assigned numer
        max_index = len(mapping) - 1

        '''
        Cant be self.indices_and_tensors because we could use the same tensor as intermediate
        '''
        new_indices_and_tensors = []

        for indices, tensor in self.indices_and_tensors:
            numeric_indices = []
            for index in indices:
                if index not in mapping:
                    mapping[index] = max_index + 1
                    max_index = mapping[index]
                numeric_indices.append(mapping[index])
            reordered_indices_and_tensors, max_index = tensor.reorder(numeric_indices, max_index)
            new_indices_and_tensors += reordered_indices_and_tensors

        return new_indices_and_tensors

class EinTen:
    slots = ['name', 'addends', 'subscripted_addends', 'converted']
    
    '''
    Todo create a class that does not need the name
    The name is only used for the input tensors
    So it is not needed for intermediate tensors
    Maybe create a base class for the tensors
    '''

    def __init__(self, name) -> None:
        self.name = name
        self.addends = [EinTenAddend(self.name)]
        self.subscripted_addends = []
        self.converted = True

    @classmethod
    def from_subscripted_addends(cls, subscripted_addends):
        instance = cls('result')
        instance.subscripted_addends = subscripted_addends
        instance.converted = False
        return instance

    def __getattr__(self, indices):
        if indices.startswith('_'):
            raise AttributeError(f"Attribute {indices} not found.")
        else:
            if not self.converted:
                self.convert_with_implicit_notation()
            return EinTen.from_subscripted_addends([a.get_subscripted(indices) for a in self.addends])

    def __setattr__(self, name, value):
        if name.startswith('_') or name in self.slots:
            super().__setattr__(name, value)
        elif isinstance(value, EinTen):
            self.addends = value.convert_subscripted_addends(list(name))
        else:
            raise ValueError(f"Assignment with type {type(value)} is not supported.")

    def evaluate(self, **kwargs):
        '''Apply the tensor to the arguments
        Construct the einsum arguments and calling np.einsum
        '''
        if not self.converted:
            self.convert_with_implicit_notation()
        result = self.addends[0].evaluate(**kwargs)
        for ten in self.addends[1:]:
            result += ten.evaluate(**kwargs)
        return result

    def print(self):
        if not self.converted:
            self.convert_with_implicit_notation()
        result = f'{self.name}_' + ''.join([chr(97 + i) for i in self.addends[0].index]) + ' = '
        space = '\n' + ' ' * (len(result) - 3) + ' + '
        return  result + space.join([ten.print() for ten in self.addends])

    def convert_with_implicit_notation(self):
        '''Convert the operations indices to numbers
        Called when the tensor has not been explicitly subscripted
        like:
        self = A.ij * B.jk
        so the indices are implicit 
        '''
        
        out_indeces = []

        # find the indices used
        for addend in self.subscripted_addends:
            counter = {}
            for indices, _ in addend.indices_and_tensors:
                for index in indices:
                    if index not in counter:
                        counter[index] = 0
                    counter[index] += 1
            out_indeces_local = [index for index, count in counter.items() if count == 1]
            out_indeces_local.sort()

            # check if they are the same for all addendus
            if out_indeces:
                if out_indeces_local != out_indeces:
                    raise ValueError(f"Indices {out_indeces_local} are not unique in the output")

            out_indeces = out_indeces_local

        self.addends = self.convert_subscripted_addends(out_indeces)
        self.converted = True

    def convert_subscripted_addends(self, indices):
        '''Convert the operations indices to numbers
        '''

        numerical_addends = []
        for addend in self.subscripted_addends:
            converted_addend = EinTenAddend(self.name)
            converted_addend.update_from_indexed(indices, addend)
            numerical_addends.append(converted_addend)

        return numerical_addends

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            new_subscripted_addends = []
            for addend in self.subscripted_addends:
                new_subscripted_addends.append(addend * other)
            result = EinTen.from_subscripted_addends(new_subscripted_addends)
        elif isinstance(other, EinTen):
            new_subscripted_addends = []
            for addend in self.subscripted_addends:
                for other_addend in other.subscripted_addends:
                    new_subscripted_addends.append((addend * other_addend))
            result = EinTen.from_subscripted_addends(new_subscripted_addends)
        else:
            raise ValueError(f"Multiplication with type {type(other)} is not supported.")

        return result

    def with_fixed(self, tensor, mode: Literal['identity', 'diagonal'] = 'identity'):
        ''' 
        Set the tensor to the identity to semplify the operations
        '''
        if mode not in ['identity', 'diagonal']:
            raise ValueError(f"Mode {mode} is not supported to fix tensors")

        if not self.converted:
            self.convert_with_implicit_notation()

        result = EinTen(f'{self.name}_{tensor.name}_identity')
        
        new_addends = []
        for addend in self.addends:
            new_addend = addend.with_fixed(tensor, mode=mode)
            new_addends.append(new_addend)

        result.addends = new_addends

        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if not isinstance(other, EinTen):
            raise ValueError(f"Addition with type {type(other)} is not supported.")
        return EinTen.from_subscripted_addends(self.subscripted_addends + other.subscripted_addends)

    def __sub__(self, other):
        if not isinstance(other, EinTen):
            raise ValueError(f"Addition with type {type(other)} is not supported.")
        return EinTen.from_subscripted_addends(self.subscripted_addends + [-1 * addend for addend in other.subscripted_addends])

    def __neg__(self):
        return EinTen.from_subscripted_addends([-1 * addend for addend in self.subscripted_addends])

#!A = EinTen('A')
#!B = EinTen('B')
#!
#!a = np.random.rand(3, 5)
#!
#!B.ij = A.ji
#!
#!np.allclose(B.evaluate(A=a), a.T)
#!
#!A = EinTen('A')
#!B = EinTen('B')
#!C = EinTen('C')
#!Z = EinTen('Z')
#!
#!a = np.random.rand(3, 3)
#!b = np.random.rand(3, 3)
#!c = np.random.rand(3, 3, 3)
#!
#!Z.kji = 2.0 * A.ij * B.jk + C.ijk
#!assert np.allclose(Z.evaluate(A=a, B=b, C=c), 2.0 * np.einsum('ij,jk->kji', a, b) + np.einsum('ijk->kji', c))
#!print(Z.print())
