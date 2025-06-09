from collections import defaultdict

import numpy as np

try:
    import opt_einsum as oe
except Exception:  # pragma: no cover - optional dependency
    oe = None


def disable_opt_einsum():
    """
    Disable the opt_einsum library if it is installed.
    This is useful for testing purposes or when you want to avoid using opt_einsum.
    """
    global oe
    oe = None


def find_root(key, mapping):
    """
    Follow the mapping chain from key and return its final representative.
    If a cycle is detected, choose a canonical representative from the cycle
    (here, the minimum key in the cycle) and update the mapping for all members.
    """
    visited = []
    current = key
    while current in mapping:
        if current in visited:
            # Cycle detected: select a representative from the cycle.
            cycle = visited[visited.index(current) :]  # the cycle part of the chain
            rep = min(cycle)  # choose canonical representative (could be any consistent choice)
            # Update mapping for all keys in the cycle.
            for node in cycle:
                mapping[node] = rep
            # Also update keys leading to the cycle.
            for node in visited:
                mapping[node] = rep
            return rep
        visited.append(current)
        current = mapping[current]
    # No cycle encountered; perform path compression.
    for node in visited:
        mapping[node] = current
    return current


def _idx_to_str(idx):
    if idx > 9:
        return chr(0x208D) + "".join(chr(0x2080 + int(i)) for i in str(idx)) + chr(0x208E)
    return chr(0x2080 + idx)


class EinTenBaseTensor:

    def __init__(self, name: str, shape, constant=None) -> None:
        if not isinstance(name, str) or name == "":
            raise ValueError(f"Name '{name}' is not valid")
        self.name = name
        self.constant = constant
        self.shape = tuple(shape)

    def __eq__(self, other):
        if not isinstance(other, EinTenBaseTensor):
            return False
        if self.name != other.name:
            return False
        if self.shape != other.shape:
            return False
        return True


class EinTenContraction:

    def __init__(self, out_indices, indices_and_tensors, prefactor=1.0, delta=None) -> None:
        # the list of indices and tensors
        self.indices_and_tensors = indices_and_tensors
        # the output indices
        self.out_indices = out_indices
        # the prefactor of the contraction
        self.prefactor = prefactor
        # the delta indices
        self.delta = {}
        if delta is not None:
            self.delta = delta
        # to cache the einsum_path
        self.einsum_path = None
        self.opt_einsum_path = None

        if self.indices_and_tensors:
            self.max_internal_index = max([max(indices) for indices, _ in self.indices_and_tensors])
        else:
            self.max_internal_index = -1

        # maps to check the equivalence of the contractions
        self._map = None
        self._detailed_map = None

    @property
    def is_base(self):
        if len(self.indices_and_tensors) != 1:
            return False
        return isinstance(self.indices_and_tensors[0][1], EinTenBaseTensor)

    def copy(self):
        return EinTenContraction(
            self.out_indices,
            self.indices_and_tensors.copy(),
            prefactor=self.prefactor,
            delta=self.delta.copy(),
        )

    def get_map(self, with_count=False):
        """
        Build and cache a map of contractions, optimized for faster grouping.
        """
        # 1. Check cache first
        if not with_count and self._map is not None:
            return self._map
        if with_count and self._detailed_map is not None:
            return self._detailed_map

        indices_and_tensors = self.indices_and_tensors
        out_indices = self.out_indices
        delta = self.delta

        # Use defaultdict for faster grouping
        contractions = defaultdict(list)

        # 1. Process output indices in O(n)
        output_positions = defaultdict(list)
        for pos, idx in enumerate(out_indices):
            output_positions[idx].append(pos)
        for idx, pos_list in output_positions.items():
            entry = ("", tuple(pos_list), 0) if with_count else ("", tuple(pos_list))
            contractions[idx].append(entry)

        # 2. Process tensor indices in O(total_indices)
        tensor_count = defaultdict(int)
        for indices, tensor in indices_and_tensors:
            name = tensor.name
            tensor_count[name] += 1
            count = tensor_count[name]
            idx_positions = defaultdict(list)
            for pos, idx in enumerate(indices):
                idx_positions[idx].append(pos)
            for idx, pos_list in idx_positions.items():
                entry = (name, tuple(pos_list), count) if with_count else (name, tuple(pos_list))
                contractions[idx].append(entry)

        # 3. Process equivalent deltas
        eq = defaultdict(list)
        for k, v in delta.items():
            root = find_root(k, delta)
            eq[root].append(k)
            eq[root].append(v)
        for delta_count, (root, items) in enumerate(eq.items()):
            for val in items:
                entry = ("_ec_delta", (val,), delta_count) if with_count else ("_ec_delta", (val,))
                contractions[root].append(entry)

        # 4. Build final sorted tuple
        grouped = [tuple(sorted(entries)) for entries in contractions.values()]
        result = tuple(sorted(grouped))
        # 6. Cache the result
        if with_count:
            self._detailed_map = result
        else:
            self._map = result

        return result

    def equal(self, other, check_prefactor=True):
        if check_prefactor and self.prefactor != other.prefactor:
            return False

        if not isinstance(other, EinTenContraction):
            return False

        if self.get_map() != other.get_map():
            return False

        return self.are_really_equivalent(other)

    def are_really_equivalent(self, other):
        detailed_map_self = self.get_map(with_count=True)
        detailed_map_other = other.get_map(with_count=True)
        """
        Check whether two detailed mappings are truly equivalent.
        Identical tensors can be swapped if they are contracted multiple times.
        So we need to check that the detailed mappings are consistent with each other.
        Let's try to find consistent assumptions for the mapping.
        """
        assumptions = {}
        for contraction1, contraction2 in zip(detailed_map_self, detailed_map_other):
            local_assumptions = {}
            for name1, idx1, id1 in contraction1:
                local_assumptions[(name1, id1)] = set()
                for name2, idx2, id2 in contraction2:
                    if name1 == name2 and idx1 == idx2:
                        local_assumptions[(name1, id1)].add(id2)
            for key, id_set in local_assumptions.items():
                if key in assumptions:
                    assumptions[key] &= id_set
                    if not assumptions[key]:
                        return False
                else:
                    assumptions[key] = id_set
        return True

    def to_string_with_subscripts(self, ss_to_idx):
        max_i_ss = 0

        idx_to_idx_for_print = {}
        for i, (_, idxs) in enumerate(ss_to_idx.items()):
            for idx in idxs:
                if idx not in idx_to_idx_for_print:
                    idx_to_idx_for_print[idx] = i
                    max_i_ss = max(max_i_ss, i)

        def idx_to_str(idx):
            if idx in idx_to_idx_for_print:
                return _idx_to_str(idx_to_idx_for_print[idx])
            nonlocal max_i_ss
            max_i_ss += 1
            idx_to_idx_for_print[idx] = max_i_ss
            return _idx_to_str(max_i_ss)

        result = f"{self.prefactor:+5f} ("
        result += "".join(
            [
                f"{ten.name}" + "".join([idx_to_str(i) for i in indices])
                for indices, ten in self.indices_and_tensors
            ]
        )
        if self.delta:
            result += "".join(f"δ{idx_to_str(k)}{idx_to_str(v)}" for k, v in self.delta.items())
        result += ")"
        result += "".join([idx_to_str(i) for i in self.out_indices])

        idx_to_ss = {idxs[0]: ss for ss, idxs in ss_to_idx.items()}
        result += (
            " {"
            + ", ".join(f"{idx_to_idx_for_print[idx]}➞{ss}" for idx, ss in idx_to_ss.items())
            + "}"
        )

        return result

    def to_string(self, ss_to_idx=None):
        if ss_to_idx is not None:
            return self.to_string_with_subscripts(ss_to_idx)

        result = f"{self.prefactor:+5f} ("
        result += "".join(
            [
                f"{ten.name}" + "".join([_idx_to_str(i) for i in indices])
                for indices, ten in self.indices_and_tensors
            ]
        )
        if self.delta:
            result += "".join(f"δ{_idx_to_str(k)}{_idx_to_str(v)}" for k, v in self.delta.items())
        result += ")"
        result += "".join([_idx_to_str(i) for i in self.out_indices])

        if ss_to_idx is not None:
            result += " " + str(ss_to_idx)

        return result

    def __repr__(self):
        return self.to_string()

    def substitute(self, old_tensor: "EinTen", *new_tensors):
        """
        Substitute a base tensor with the new tensors
        """
        if not old_tensor.is_base:
            raise ValueError(f"Tensor {old_tensor} is not a base tensor")

        old_prefactor = old_tensor.addends[0].prefactor
        old_tensor = old_tensor.addends[0].indices_and_tensors[0][1]

        new_addends = [
            EinTenContraction(self.out_indices, [], prefactor=self.prefactor, delta=self.delta)
        ]
        for indices, ten in self.indices_and_tensors:
            if ten == old_tensor:

                new_new_addends = []
                for new_tensor in new_tensors:

                    if not new_tensor.is_base:
                        raise ValueError(f"Tensor {new_tensor} is not a base tensor")

                    new_prefactor = new_tensor.addends[0].prefactor
                    new_tensor = new_tensor.addends[0].indices_and_tensors[0][1]

                    for addend in new_addends:
                        tmp = EinTenContraction(
                            self.out_indices,
                            addend.indices_and_tensors.copy(),
                            prefactor=addend.prefactor,
                            delta=addend.delta,
                        )
                        tmp.indices_and_tensors.append((indices, new_tensor))
                        tmp.prefactor *= new_prefactor / old_prefactor
                        new_new_addends.append(tmp)

                new_addends = new_new_addends

            else:
                for addend in new_addends:
                    addend.indices_and_tensors.append((indices, ten))

        return new_addends

    def simplify(self):
        # Reduce the number of the internal indices
        # the internal indices are the out_indices and the delta indices
        # Delta is necessary to keep in the order or the out indices for subsequential assignments
        # for instance A.ijj -= A.ijj will create a delta
        # if after i use A.ijk i should remember that k is in the delta (because the out indices are i j)
        # TODO: refix out_indices to be always in the reasonable order (0 1 2 3 ...)

        internal_indices = set(self.out_indices) | set(self.delta) | set(self.delta.values())

        old_to_new_internal_indices = {}
        i = 0
        for idxs, _ in self.indices_and_tensors:
            for idx in idxs:
                if idx not in internal_indices and idx not in old_to_new_internal_indices:

                    while i in internal_indices:
                        i += 1

                    old_to_new_internal_indices[idx] = i
                    i += 1

        new_indices_and_tensors = []
        for indices, tensor in self.indices_and_tensors:
            new_indices = tuple([old_to_new_internal_indices.get(idx, idx) for idx in indices])
            new_indices_and_tensors.append((new_indices, tensor))
        self.indices_and_tensors = new_indices_and_tensors

        new_delta = {}
        for k, v in self.delta.items():
            new_delta[old_to_new_internal_indices.get(k, k)] = old_to_new_internal_indices.get(v, v)
        self.delta = new_delta

    def set_as_diagonal(self, base_tensor, to_identity=False):
        """
        Set the base_tensor to the identity to semplify the operations
        """
        if not base_tensor.is_base:
            raise ValueError(f"Tensor {base_tensor} is not a base tensor")
        base_prefactor = base_tensor.addends[0].prefactor
        base_tensor = base_tensor.addends[0].indices_and_tensors[0][1]

        if len(base_tensor.shape) != 2 or base_tensor.shape[0] != base_tensor.shape[1]:
            raise ValueError(
                f"Tensor {base_tensor} is not a matrix or is not square, shape {base_tensor.shape}"
            )
        new_shape = (base_tensor.shape[0],)
        new_base_tensor = EinTenBaseTensor(base_tensor.name, new_shape, base_tensor.constant)

        prefactor = self.prefactor
        indices_and_tensors = []
        new_delta = {}

        for indices, ten in self.indices_and_tensors:
            if ten == base_tensor:

                prefactor /= base_prefactor

                new_delta[max(indices)] = min(indices)

                if not to_identity:
                    indices_and_tensors.append(([indices[0]], new_base_tensor))

            else:
                indices_and_tensors.append((indices, ten))

        print("TODO: to generalize the new_delta update")
        new_delta = new_delta | self.delta

        new_indices_and_tensors = []
        for indices, ten in indices_and_tensors:
            indices = [new_delta.get(i, i) for i in indices]
            new_indices_and_tensors.append((indices, ten))
        indices_and_tensors = new_indices_and_tensors

        print("TODO: to generalize the out_indices update")
        out_indices = [new_delta.get(i, i) for i in self.out_indices]

        return EinTenContraction(
            out_indices, indices_and_tensors, prefactor=prefactor, delta=new_delta
        )

    def evaluate(self, memory_limit=None, **kwargs):
        if oe is None:
            return self.evaluate_numpy(**kwargs)
        else:
            return self.evaluate_opt_einsum(memory_limit=memory_limit, **kwargs)

    def evaluate_opt_einsum(self, memory_limit=None, **kwargs):
        # Construct the einsum arguments
        # args = [(ten.shape, indices) for indices, ten in self.indices_and_tensors]
        constants = []
        contract_expression_args = []
        args = []
        for i, (indices, ten) in enumerate(self.indices_and_tensors):
            if ten.constant is not None:
                constants.append(i)
                contract_expression_args.append(ten.constant)
                contract_expression_args.append(indices)
            else:
                contract_expression_args.append(ten.shape)
                contract_expression_args.append(indices)
                args.append(kwargs[ten.name])

        # Flatten the list
        contract_expression_args.append(self.out_indices)
        return self.prefactor * self.get_opt_einsum_path(
            contract_expression_args, constants, memory_limit=memory_limit
        )(*args)

    def get_opt_einsum_path(self, args, constants, memory_limit=None):
        if self.opt_einsum_path is None:
            self.opt_einsum_path = oe.contract_expression(
                *args, constants=constants, memory_limit=memory_limit
            )
        return self.opt_einsum_path

    def evaluate_numpy(self, **kwargs):
        """Apply the tensor to the arguments
        Construct the einsum arguments and calling np.einsum
        """

        # Construct the einsum arguments
        args = [(kwargs[ten.name], indices) for indices, ten in self.indices_and_tensors]
        # Flatten the list
        args = [arg for pair in args for arg in pair]

        if len(set(self.out_indices)) != len(self.out_indices):

            """
            We have to first create a writeable view with einsum
            ex: ijkijk->ijk create a view with ijk and then assign the result
            """

            idx_shape = {}
            for idx in self.out_indices:

                for indices, ten in self.indices_and_tensors:
                    for i, index in enumerate(indices):
                        if index == idx:
                            idx_shape[idx] = kwargs[ten.name].shape[i]

            result_shape = []
            for idx in self.out_indices:
                result_shape.append(idx_shape[idx])

            unique_indices = list(set(self.out_indices))

            args.append(unique_indices)
            result_tmp = self.prefactor * np.einsum(*args, optimize=self.get_einpath(args))

            # copy the result in the right place
            result = np.zeros(result_shape, dtype=result_tmp.dtype)

            # this create a view and copy the result
            result_view = np.einsum(result, self.out_indices, unique_indices)
            result_view[:] = result_tmp

            return result

            #
            #   Alternative explicit way
            #

        #'''
        # Indices are not unique in the output
        # Einsum does not allow repeated indices
        #'''
        # map_idx = {}
        # for i, idx in enumerate(self.index):
        #    map_idx[idx] = map_idx.get(idx, []) + [i]

        ## take only the unique
        # args.append([idx for idx in map_idx])

        # if self.einsum_path is None:
        #    self.einsum_path, _ = np.einsum_path(*args, optimize=True)

        # result = self.prefactor * np.einsum(*args, optimize=self.einsum_path)

        #'''
        # We have to unpack the repeated indices, one at time
        # We create a tmp variable to store the position of the indices in the tmp array
        # the packed indices are replaced with None
        #'''

        # tmp_idx_with_none = [None] * len(self.index)
        # for idx, i in map_idx.items():
        #    tmp_idx_with_none[i[0]] = idx

        # shape_idx = {}
        # for i, idx in enumerate(map_idx):
        #    shape_idx[idx] = result.shape[i]

        # for idx, i in map_idx.items():
        #    # lets unpack one repeated index at a time

        #    if len(i) == 1:
        #        continue

        #    tmp_slice = []
        #    for tmp_idx in tmp_idx_with_none:
        #        if tmp_idx is None:
        #            continue

        #        if tmp_idx != idx:
        #            tmp_slice.append(slice(None))
        #        else:
        #            tmp_slice.append(np.arange(shape_idx[idx]))

        #    for i_ in map_idx[idx]:
        #        tmp_idx_with_none[i_] = idx

        #    target_slice = []
        #    target_shape = []
        #    for tmp_idx in tmp_idx_with_none:
        #        if tmp_idx is None:
        #            continue

        #        target_shape.append(shape_idx[tmp_idx])

        #        if tmp_idx != idx:
        #            target_slice.append(slice(None))
        #        else:
        #            target_slice.append(np.arange(shape_idx[tmp_idx]))

        #    target = np.zeros(target_shape, dtype=result.dtype)
        #    target[tuple(target_slice)] = result[tuple(tmp_slice)]

        #    result = target

        # return result

        else:
            args.append(self.out_indices)
            return self.prefactor * np.einsum(*args, optimize=self.get_einpath(args))

    def get_einpath(self, args):
        if self.einsum_path is None:
            try:
                self.einsum_path, _ = np.einsum_path(*args, optimize=True)
            except ValueError:
                raise ValueError(f"Error in einsum_path for {self}")
        return self.einsum_path

    def reduce(
        self,
        old_to_new_indices,
        out_indices=None,
        delta=None,
        start_internal_index=None,
    ):
        """
        Perform a reduction of the subscripts
        old_to_new_indices is a dictionary that maps the old indices to the new ones
        """

        local_old_to_new_indices = old_to_new_indices.copy()

        max_internal_index = start_internal_index
        if max_internal_index is None:
            max_internal_index = self.max_internal_index

        # updating delta with the new indices
        # there could be a case where the new index is already in the delta
        # for instance δ_ij * δ_ik -> δ_ij * δ_kj so in the end all becomes j
        new_delta = {}
        for k, v in self.delta.items():
            if k not in local_old_to_new_indices:
                max_internal_index += 1
                local_old_to_new_indices[k] = max_internal_index
            if v not in local_old_to_new_indices:
                max_internal_index += 1
                local_old_to_new_indices[v] = max_internal_index
            new_k = local_old_to_new_indices[k]
            new_v = local_old_to_new_indices[v]

            if new_k in new_delta:
                if new_delta[new_k] != new_v:
                    new_delta[new_v] = new_delta[new_k]
            else:
                # could be already in the dictionary
                # new_delta[new_k] = new_delta.get(new_v, new_v)
                new_delta[new_k] = new_v

        # First, update new_delta with the provided delta values.
        if delta is not None:
            for k, v in delta.items():
                if k in new_delta:
                    new_delta[v] = new_delta[k]
                else:
                    new_delta[k] = v

        # Resolve cycles (and perform path compression) in new_delta.
        for k in list(new_delta.keys()):
            new_delta[k] = find_root(k, new_delta)

        new_indices_and_tensors = []
        for indices, tensor in self.indices_and_tensors:
            new_indices = []
            for i in indices:

                # if the index is not in the dictionary we have to assign a new index
                if i not in local_old_to_new_indices:
                    max_internal_index += 1
                    local_old_to_new_indices[i] = max_internal_index

                # We have to take in account the delta indices
                if local_old_to_new_indices[i] in new_delta:
                    new_indices.append(new_delta[local_old_to_new_indices[i]])
                else:
                    new_indices.append(local_old_to_new_indices[i])

            new_indices_and_tensors.append((new_indices, tensor))

        if out_indices is None:
            out_indices = tuple([local_old_to_new_indices[i] for i in self.out_indices])
        # apply the delta to the out_indices
        out_indices = tuple(new_delta.get(i, i) for i in out_indices)

        return EinTenContraction(
            out_indices,
            new_indices_and_tensors,
            prefactor=self.prefactor,
            delta=new_delta,
        )

    def merge(self, other):
        if isinstance(other, EinTenContraction):
            new_indices_and_tensors = self.indices_and_tensors + other.indices_and_tensors
            new_prefactor = self.prefactor * other.prefactor
            new_delta = self.delta | other.delta
            new_out_indices = self.out_indices + other.out_indices
            return EinTenContraction(
                new_out_indices,
                new_indices_and_tensors,
                prefactor=new_prefactor,
                delta=new_delta,
            )
        else:
            raise ValueError(f"Multiplication with type {type(other)} is not supported.")

    def __mul__(self, other):
        return EinTenContraction(
            self.out_indices,
            self.indices_and_tensors,
            delta=self.delta,
            prefactor=self.prefactor * other,
        )

    def pop(self, base_tensor):

        if not base_tensor.is_base:
            raise ValueError(f"Tensor {base_tensor} is not a base tensor")

        base_tensor = base_tensor.addends[0].indices_and_tensors[0][1]

        indices_and_tensors = []
        indices = []
        for idx, ten in self.indices_and_tensors:
            if ten == base_tensor:
                indices.append(idx)
            else:
                indices_and_tensors.append((idx, ten))
        self.indices_and_tensors = indices_and_tensors
        return indices


class EinTen:
    slots = ["addends", "ss_to_idx"]

    """
    Todo create a class that does not need the name
    The name is only used for the input tensors
    So it is not needed for intermediate tensors
    Maybe create a base class for the tensors
    """

    def __init__(self, name, shape, constant=None) -> None:
        # Create a base tensor
        base_tensor = EinTenBaseTensor(name, shape, constant)
        indices = tuple(range(len(shape)))
        # self.addends = [(indices, (), EinTenContraction([[indices, base_tensor]]))]
        self.addends = [EinTenContraction(indices, [[indices, base_tensor]])]
        self.ss_to_idx = {}

    @classmethod
    def empty(cls):
        einten = object.__new__(EinTen)
        einten.addends = []
        einten.ss_to_idx = {}
        return einten

    @property
    def is_base(self):
        return len(self.addends) == 1 and self.addends[0].is_base

    def get_max_idx(self):
        if len(self.addends) == 0:
            return len(self.ss_to_idx)
        return max([a.max_internal_index for a in self.addends])

    @classmethod
    def from_contraction_list(cls, addends, ss_to_idx=None):
        einten = object.__new__(EinTen)
        einten.addends = addends
        if ss_to_idx is not None:
            einten.ss_to_idx = ss_to_idx
            # einten.clean_ss_to_idx()
        return einten

    def clean_ss_to_idx(self):
        # Collect all internal indices into a set for faster lookups
        internal_indices = set()
        for addend in self.addends:
            for indices, _ in addend.indices_and_tensors:
                internal_indices.update(indices)

        # Filter ss_to_idx based on internal_indices
        new_ss_to_idx = {}
        for ss, idxs in self.ss_to_idx.items():
            filtered = [idx for idx in idxs if idx in internal_indices]
            if filtered:
                new_ss_to_idx[ss] = filtered

        self.ss_to_idx = new_ss_to_idx

    def to_implicit_notation(self):
        implicit_ss = None

        # check if its the same for all the addends
        for a in self.addends:

            tmp_ss = {}
            for ss in self.ss_to_idx:
                for idx in a.out_indices:
                    if idx in self.ss_to_idx[ss]:
                        tmp_ss[ss] = tmp_ss.get(ss, 0) + 1

            tmp_ss = [ss for ss, count in tmp_ss.items() if count == 1]
            tmp_ss.sort()
            tmp_ss = "".join(tmp_ss)

            if implicit_ss is None:
                implicit_ss = tmp_ss

            if implicit_ss != tmp_ss:
                raise ValueError(f"Subscripts are not unique in the output")

        if implicit_ss is None:
            implicit_ss = ""

        self.assign(implicit_ss, self)

    def __eq__(self, other):
        if not isinstance(other, EinTen):
            return False
        if len(self.ss_to_idx) != 0:
            self.to_implicit_notation()
        if len(other.ss_to_idx) != 0:
            other.to_implicit_notation()
        if len(self.addends) != len(other.addends):
            return False

        other_to_check = [i for i in range(len(other.addends))]
        self_to_check = [i for i in range(len(self.addends))]

        for i_a, a in enumerate(self.addends):

            found = False

            for i_b, b in enumerate(other.addends):
                if i_b in other_to_check and a.equal(b):
                    other_to_check.remove(i_b)
                    found = True
                    break

            if not found:
                return False

            self_to_check.remove(i_a)

        return len(other_to_check) == 0 and len(self_to_check) == 0

    def __repr__(self):
        if self.ss_to_idx:
            return "\n".join([a.to_string(ss_to_idx=self.ss_to_idx) for a in self.addends])
        return "\n".join([a.to_string() for a in self.addends])

    def __getattr__(self, subscripts) -> "EinTen":
        if subscripts.startswith("_"):
            raise AttributeError(f"Attribute {subscripts} not found.")
        else:
            if isinstance(subscripts, str):
                subscripts = tuple(subscripts)
            return self.get_subscripted(subscripts)

    def __getitem__(self, subscripts) -> "EinTen":
        if isinstance(subscripts, str):
            subscripts = tuple(subscripts)
        elif not isinstance(subscripts, tuple):
            subscripts = (subscripts,)
        return self.get_subscripted(subscripts)

    def get_subscripted(self, subscripts):

        if len(self.ss_to_idx) != 0:
            self.assign(subscripts, self)
            self.ss_to_idx = {}
            return self.get_subscripted(sorted(subscripts))

        if self.addends and len(subscripts) != len(self.addends[0].out_indices):
            expected_length = len(self.addends[0].out_indices)
            actual_length = len(subscripts)
            raise ValueError(
                f"Invalid subscript for tensor {self}:\n"
                f"  Expected {expected_length} indices "
                f"but received {actual_length} indices: {subscripts}"
            )

        ss_to_idx = {}
        for idx, ss in enumerate(subscripts):
            ss_to_idx[ss] = ss_to_idx.get(ss, []) + [idx]
        return EinTen.from_contraction_list(self.addends, ss_to_idx=ss_to_idx)

    def __setattr__(self, subscripts, value):
        if subscripts.startswith("_") or subscripts in self.slots:
            return super().__setattr__(subscripts, value)
        if isinstance(subscripts, str):
            subscripts = tuple(subscripts)
        self.assign(subscripts, value)

    def __setitem__(self, subscripts, value):
        if isinstance(subscripts, str):
            subscripts = tuple(subscripts)
        elif not isinstance(subscripts, tuple):
            subscripts = (subscripts,)
        self.assign(subscripts, value)

    def assign(self, subscripts, other):
        if not isinstance(other, EinTen):
            raise ValueError(f"Assignment with type {type(other)} is not supported.")

        subs_index = {ss: i for i, ss in enumerate(subscripts)}

        max_current_index = max(self.get_max_idx(), max(subs_index.values(), default=-1))

        # build the mapping from other's output indices to the target ones
        other_to_self_indices = {}
        for ss in subscripts:
            for idx in other.ss_to_idx[ss]:
                other_to_self_indices[idx] = subs_index[ss]

        for ss, idxs in other.ss_to_idx.items():
            if ss not in subscripts:
                max_current_index += 1
                for idx in idxs:
                    other_to_self_indices[idx] = max_current_index

        new_indices = tuple(subs_index[ss] for ss in subscripts)

        # Check if we can directly copy the addends without reducing them
        if (
            len(set(subscripts)) == len(subscripts)
            and set(other.ss_to_idx.keys()) == set(subscripts)
            and all(
                len(other.ss_to_idx[ss]) == 1 and other.ss_to_idx[ss][0] == subs_index[ss]
                for ss in subscripts
            )
            and all(a.out_indices == new_indices and not a.delta for a in other.addends)
        ):
            self.addends = [a.copy() for a in other.addends]
            self.ss_to_idx = {}
            self.simplify()
            return

        # from here we need to actually reduce each addend

        if len(set(subscripts)) != len(subscripts):
            # We are updating a diagonal element

            # this impose that delta is always greater to lowert
            # δ_ij => i > j
            delta = {}
            for i, ss in enumerate(subscripts):
                if subscripts.index(ss) != i:
                    delta[i] = subscripts.index(ss)

            # first we have to set to zero the diagonal element
            # diag = - self.__getattr__(subscripts)
            diag = -self.get_subscripted(subscripts)

            diag_to_self_indices = {}
            for ss in subscripts:
                for idx in diag.ss_to_idx[ss]:
                    diag_to_self_indices[idx] = subscripts.index(ss)

            addends = []

            for addend in diag.addends:
                addends.append(
                    addend.reduce(diag_to_self_indices, out_indices=new_indices, delta=delta)
                )

            for addend in other.addends:
                addends.append(
                    addend.reduce(other_to_self_indices, out_indices=new_indices, delta=delta)
                )

            self.addends += addends
        else:
            addends = []

            for addend in other.addends:
                addends.append(addend.reduce(other_to_self_indices, out_indices=new_indices))

            self.addends = addends

        self.ss_to_idx = {}
        self.simplify()

    @classmethod
    def quick_sum(cls, eintens):
        """Efficiently sum multiple :class:`EinTen` objects at once.

        This avoids repeated shifting of indices when adding tensors
        sequentially.  Indices for each term are shifted by a precomputed
        offset based on the maximum internal index of the preceding terms.
        """

        if not eintens:
            return cls.empty()

        offsets = []
        current = 0
        max_indices = []
        for e in eintens:
            if not isinstance(e, EinTen):
                raise ValueError(f"Addition with type {type(e)} is not supported.")
            offsets.append(current)
            max_idx = e.get_max_idx()
            max_indices.append(max_idx)
            current += max_idx + 1

        addends = []
        ss_to_idx = {}

        for einten, start_other_index, max_idx in zip(eintens, offsets, max_indices):
            start_internal_other_index = start_other_index + max_idx
            other_old_to_new_idx = {}
            for ss, idxs in einten.ss_to_idx.items():
                for idx in idxs:
                    other_old_to_new_idx[idx] = start_other_index + idx

            for a in einten.addends:
                addends.append(
                    a.reduce(
                        other_old_to_new_idx,
                        start_internal_index=start_internal_other_index,
                    )
                )

            for ss, idxs in einten.ss_to_idx.items():
                for idx in idxs:
                    ss_to_idx[ss] = ss_to_idx.get(ss, []) + [start_other_index + idx]

        return cls.from_contraction_list(addends, ss_to_idx=ss_to_idx)

    def simplify(self):

        simple_map = {}
        order = []

        for addend in self.addends:
            key = addend.get_map()
            if key not in simple_map:
                simple_map[key] = [addend]
                order.append(key)
            else:
                for existing in simple_map[key]:
                    if existing.equal(addend, check_prefactor=False):
                        existing.prefactor += addend.prefactor
                        break
                else:
                    simple_map[key].append(addend)

        new_addends = []
        for key in order:
            for addend in simple_map[key]:
                if addend.prefactor != 0.0:
                    addend.simplify()
                    new_addends.append(addend)

        # if the new_addends is empty, we add one to ensure the tensor is not empty
        if len(new_addends) == 0:
            for key in order:
                for addend in simple_map[key]:
                    new_addends.append(addend)
                    break
                if len(new_addends) > 0:
                    break

        self.addends = new_addends

    def evaluate(self, memory_limit=None, **kwargs):
        """Apply the tensor to the arguments
        Construct the einsum arguments and calling np.einsum
        """
        if len(self.ss_to_idx) != 0:
            self.to_implicit_notation()
        if oe is None:
            result = self.addends[0].evaluate(memory_limit=memory_limit, **kwargs)
            for a in self.addends[1:]:
                result += a.evaluate(memory_limit=memory_limit, **kwargs)
            return result
        else:
            with oe.shared_intermediates():
                result = self.addends[0].evaluate(memory_limit=memory_limit, **kwargs)
                for a in self.addends[1:]:
                    result += a.evaluate(memory_limit=memory_limit, **kwargs)
                return result

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return EinTen.from_contraction_list([a * other for a in self.addends], self.ss_to_idx)
        elif isinstance(other, EinTen):

            if len(self.ss_to_idx) == 0 or len(other.ss_to_idx) == 0:
                raise ValueError(f"Cannot multiply not subscripted tensors: {self} and {other}")

            # the other indices are the shifted by the maximum index of self
            start_other_index = self.get_max_idx() + 1

            other_old_to_new_idx = {}
            other_ss_to_idx = {}
            start_internal_other_index = start_other_index
            for ss, idxs in other.ss_to_idx.items():
                for idx in idxs:
                    other_old_to_new_idx[idx] = start_other_index + idx
                    other_ss_to_idx[ss] = other_old_to_new_idx[idx]
                    start_internal_other_index = max(
                        start_internal_other_index, start_other_index + idx
                    )

            new_other_addends = []
            for a in other.addends:
                new_other_addends.append(
                    a.reduce(
                        other_old_to_new_idx,
                        start_internal_index=start_internal_other_index,
                    )
                )

            new_addends = []
            for a1 in self.addends:
                for a2 in new_other_addends:
                    new_addends.append(a1.merge(a2))

            # now we have to update the ss_to_idx
            ss_to_idx = {ss: idxs.copy() for ss, idxs in self.ss_to_idx.items()}

            for ss, idxs in other.ss_to_idx.items():
                for idx in idxs:
                    ss_to_idx[ss] = ss_to_idx.get(ss, []) + [start_other_index + idx]

            return EinTen.from_contraction_list(new_addends, ss_to_idx=ss_to_idx)

        else:
            raise ValueError(f"Multiplication with type {type(other)} is not supported.")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if other == 0:
            # TODO mayba do a copy
            return self
        if isinstance(other, EinTen):
            # if self.subscripts is None or other.subscripts is None:
            #    raise ValueError(f"Cannot add not subscripted tensors: {self} and {other}")

            # the other indices are the shifted by the maximum index of self
            start_other_index = self.get_max_idx() + 1

            start_internal_other_index = start_other_index
            other_old_to_new_idx = {}
            for ss, idxs in other.ss_to_idx.items():
                for idx in idxs:
                    other_old_to_new_idx[idx] = start_other_index + idx
                    start_internal_other_index = max(
                        start_internal_other_index, start_other_index + idx
                    )

            new_other_addends = []
            for a in other.addends:
                new_other_addends.append(
                    a.reduce(
                        other_old_to_new_idx,
                        start_internal_index=start_internal_other_index,
                    )
                )

            addends = self.addends + new_other_addends

            ss_to_idx = {ss: idxs.copy() for ss, idxs in self.ss_to_idx.items()}
            for ss, idxs in other.ss_to_idx.items():
                for idx in idxs:
                    ss_to_idx[ss] = ss_to_idx.get(ss, []) + [start_other_index + idx]

            return EinTen.from_contraction_list(addends, ss_to_idx=ss_to_idx)

        else:
            raise ValueError(f"Addition with type {type(other)} is not supported.")

    def sum(self, eintens):

        addends = self.addends
        ss_to_idx = {ss: idxs.copy() for ss, idxs in self.ss_to_idx.items()}
        start_other_index = self.get_max_idx() + 1

        for einten in eintens:

            start_internal_other_index = start_other_index
            other_old_to_new_idx = {}
            for ss, idxs in einten.ss_to_idx.items():
                for idx in idxs:
                    other_old_to_new_idx[idx] = start_other_index + idx
                    start_internal_other_index = max(
                        start_internal_other_index, start_other_index + idx
                    )

            new_other_addends = []
            for a in einten.addends:
                new_addend = a.reduce(
                    other_old_to_new_idx,
                    start_internal_index=start_internal_other_index,
                )
                start_other_index = max(start_other_index, new_addend.max_internal_index)
                new_other_addends.append(new_addend)

            addends += new_other_addends

            ss_to_idx = {ss: idxs.copy() for ss, idxs in self.ss_to_idx.items()}
            for ss, idxs in einten.ss_to_idx.items():
                for idx in idxs:
                    ss_to_idx[ss] = ss_to_idx.get(ss, []) + [start_other_index + idx]

        return EinTen.from_contraction_list(addends, ss_to_idx=ss_to_idx)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise ValueError(f"Power with type {type(n)} is not supported.")
        if n == 0:
            return 1.0
        if n == 1:
            return self
        if n < 0:
            raise ValueError(f"Power < 0 is not supported")
        return self * (self ** (n - 1))

    def substitute(self, tensor, *tensors):
        """
        Substitute a base tensor with the new tensors
        """
        addends = []
        for addend in self.addends:
            addends += addend.substitute(tensor, *tensors)
        self.addends = addends
        self.simplify()

    def set_as_diagonal(self, base_tensor, to_identity=False):
        """
        Set the base tensor to the identity to simplify the operations of the contraction.
        **Note**: the rank of the contraction is never reduced
        """
        self.addends = [a.set_as_diagonal(base_tensor, to_identity) for a in self.addends]

    def get_addends(self):
        result = []
        for addend in self.addends:
            addend_copy = addend.copy()
            addend_copy.prefactor = 1.0
            result.append(
                (
                    addend.prefactor,
                    EinTen.from_contraction_list([addend_copy], ss_to_idx=self.ss_to_idx),
                )
            )
        return result

    def pop(self, base_tensor):
        """
        **Note**: for advanced usage
        """

        idx_to_ss = {idx: ss for ss, idxs in self.ss_to_idx.items() for idx in idxs}

        result = []
        for addend in self.addends:
            for idxs in addend.pop(base_tensor):
                result.append([idx_to_ss[idx] for idx in idxs])

        self.clean_ss_to_idx()

        return result
