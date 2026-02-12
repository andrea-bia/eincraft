from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np

from eincraft import utils
from eincraft.utils import find_root, _idx_to_str
from eincraft.symbol import EinTenBaseTensor


class EinTenContraction:
    """
    Represents a single term in a tensor expression, consisting of a product of tensors
    (contracted over certain indices) multiplied by a scalar prefactor.

    Mathematically, it represents something like:
        prefactor * T1_{indices1} * T2_{indices2} * ... * delta_{d1,d2} ...
    """

    def __init__(
        self,
        out_indices: Tuple[int, ...],
        indices_and_tensors: List[Tuple[Any, EinTenBaseTensor]],
        prefactor: float = 1.0,
        delta: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Initialize an EinTenContraction.

        Args:
            out_indices: The indices of the resulting tensor.
            indices_and_tensors: A list of (indices, tensor) tuples.
            prefactor: The scalar multiplier.
            delta: A dictionary representing Kronecker deltas (idx1 -> idx2 implies delta_{idx1, idx2}).
        """
        # the list of indices and tensors
        self.indices_and_tensors = indices_and_tensors
        # the output indices
        self.out_indices = out_indices
        # the prefactor of the contraction
        self.prefactor = prefactor
        # the delta indices
        self.delta: Dict[int, int] = {}
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
        self._map: Optional[Tuple[Tuple[Any, ...], ...]] = None
        self._detailed_map: Optional[Tuple[Tuple[Any, ...], ...]] = None

    @property
    def is_base(self) -> bool:
        """Check if this contraction wraps a single base tensor without contraction."""
        if len(self.indices_and_tensors) != 1:
            return False
        return isinstance(self.indices_and_tensors[0][1], EinTenBaseTensor)

    def copy(self) -> "EinTenContraction":
        """Create a deep copy of the contraction."""
        return EinTenContraction(
            self.out_indices,
            self.indices_and_tensors.copy(),
            prefactor=self.prefactor,
            delta=self.delta.copy(),
        )

    def get_map(self, with_count: bool = False) -> Tuple[Tuple[Any, ...], ...]:
        """
        Build and cache a map of contractions, optimized for faster grouping/comparison.
        This map characterizes the structure of the contraction graph.

        Args:
            with_count: If True, includes counts in the map key for more detailed comparison.
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
                entry = (
                    (name, tuple(pos_list), count) if with_count else (name, tuple(pos_list))
                )
                contractions[idx].append(entry)

        # 3. Process equivalent deltas
        eq = defaultdict(list)
        for k, v in delta.items():
            root = find_root(k, delta)
            eq[root].append(k)
            eq[root].append(v)
        for delta_count, (root, items) in enumerate(eq.items()):
            for val in items:
                entry = (
                    ("_ec_delta", (val,), delta_count)
                    if with_count
                    else ("_ec_delta", (val,))
                )
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

    def equal(self, other: Any, check_prefactor: bool = True) -> bool:
        """Check if two contractions are equivalent."""
        if check_prefactor and self.prefactor != other.prefactor:
            return False

        if not isinstance(other, EinTenContraction):
            return False

        if self.get_map() != other.get_map():
            return False

        return self.are_really_equivalent(other)

    def are_really_equivalent(self, other: "EinTenContraction") -> bool:
        """
        Check whether two detailed mappings are truly equivalent.
        Identical tensors can be swapped if they are contracted multiple times.
        """
        detailed_map_self = self.get_map(with_count=True)
        detailed_map_other = other.get_map(with_count=True)

        assumptions: Dict[Tuple[str, int], Set[int]] = {}
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

    def to_string_with_subscripts(self, ss_to_idx: Dict[str, List[int]]) -> str:
        """String representation using provided subscript mapping."""
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
            result += "".join(
                f"δ{idx_to_str(k)}{idx_to_str(v)}" for k, v in self.delta.items()
            )
        result += ")"
        result += "".join([idx_to_str(i) for i in self.out_indices])

        idx_to_ss = {idxs[0]: ss for ss, idxs in ss_to_idx.items()}
        result += (
            " {"
            + ", ".join(
                f"{idx_to_idx_for_print[idx]}➞{ss}" for idx, ss in idx_to_ss.items()
            )
            + "}"
        )

        return result

    def to_string(self, ss_to_idx: Optional[Dict[str, List[int]]] = None) -> str:
        """Convert the contraction to a string representation."""
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
            result += "".join(
                f"δ{_idx_to_str(k)}{_idx_to_str(v)}" for k, v in self.delta.items()
            )
        result += ")"
        result += "".join([_idx_to_str(i) for i in self.out_indices])

        if ss_to_idx is not None:
            result += " " + str(ss_to_idx)

        return result

    def __repr__(self) -> str:
        return self.to_string()

    def substitute(self, old_tensor: "Any", *new_tensors: "Any") -> List["EinTenContraction"]:
        """
        Substitute a base tensor with new tensors.
        """
        if not old_tensor.is_base:
            raise ValueError(f"Tensor {old_tensor} is not a base tensor")

        old_prefactor = old_tensor.addends[0].prefactor
        old_tensor_base = old_tensor.addends[0].indices_and_tensors[0][1]

        new_addends = [
            EinTenContraction(
                self.out_indices, [], prefactor=self.prefactor, delta=self.delta
            )
        ]
        for indices, ten in self.indices_and_tensors:
            if ten == old_tensor_base:

                new_new_addends = []
                for new_tensor in new_tensors:

                    if not new_tensor.is_base:
                        raise ValueError(f"Tensor {new_tensor} is not a base tensor")

                    new_prefactor = new_tensor.addends[0].prefactor
                    new_tensor_base = new_tensor.addends[0].indices_and_tensors[0][1]

                    for addend in new_addends:
                        tmp = EinTenContraction(
                            self.out_indices,
                            addend.indices_and_tensors.copy(),
                            prefactor=addend.prefactor,
                            delta=addend.delta,
                        )
                        tmp.indices_and_tensors.append((indices, new_tensor_base))
                        tmp.prefactor *= new_prefactor / old_prefactor
                        new_new_addends.append(tmp)

                new_addends = new_new_addends

            else:
                for addend in new_addends:
                    addend.indices_and_tensors.append((indices, ten))

        return new_addends

    def simplify(self) -> None:
        """
        Simplify the contraction by re-indexing internal indices to be contiguous,
        and normalizing output indices.
        """
        # Determine current used indices
        # internal_indices includes out_indices and delta indices to avoid collisions during remapping
        internal_indices = (
            set(self.out_indices) | set(self.delta) | set(self.delta.values())
        )

        old_to_new_internal_indices = {}
        
        # 1. Normalize output indices to be 0, 1, 2... if possible, 
        # or at least mapped consistently.
        # Ideally, we map out_indices[0] -> 0, out_indices[1] -> 1, etc.
        # But we must respect delta constraints.
        
        # Current implementation keeps out_indices fixed and only compacts summation indices.
        # To fully normalize (refix out_indices), we would map all indices.
        
        # Let's map out_indices first to ensure they are 0..N-1
        # This fixes the TODO: "refix out_indices to be always in the reasonable order"
        
        current_new_idx = 0
        
        # Map output indices first
        for idx in self.out_indices:
             if idx not in old_to_new_internal_indices:
                 old_to_new_internal_indices[idx] = current_new_idx
                 current_new_idx += 1
                 
        # Map delta indices
        # If an index in delta is already mapped (because it's an output index), use that.
        # Otherwise assign new index.
        sorted_delta_keys = sorted(list(set(self.delta.keys()) | set(self.delta.values())))
        for idx in sorted_delta_keys:
            if idx not in old_to_new_internal_indices:
                old_to_new_internal_indices[idx] = current_new_idx
                current_new_idx += 1

        # Map remaining internal (summation) indices
        for idxs, _ in self.indices_and_tensors:
            for idx in idxs:
                if idx not in old_to_new_internal_indices:
                    old_to_new_internal_indices[idx] = current_new_idx
                    current_new_idx += 1

        # Apply mapping to indices_and_tensors
        new_indices_and_tensors = []
        for indices, tensor in self.indices_and_tensors:
            new_indices = tuple(
                [old_to_new_internal_indices[idx] for idx in indices]
            )
            new_indices_and_tensors.append((new_indices, tensor))
        self.indices_and_tensors = new_indices_and_tensors

        # Apply mapping to delta
        new_delta = {}
        for k, v in self.delta.items():
            new_k = old_to_new_internal_indices[k]
            new_v = old_to_new_internal_indices[v]
            new_delta[new_k] = new_v
        self.delta = new_delta
        
        # Apply mapping to out_indices
        self.out_indices = tuple([old_to_new_internal_indices[idx] for idx in self.out_indices])
        
        # Re-calculate max index
        if self.indices_and_tensors:
             self.max_internal_index = max([max(indices) for indices, _ in self.indices_and_tensors])
        else:
             self.max_internal_index = -1

    def set_as_diagonal(
        self, base_tensor: Any, to_identity: bool = False, indices: Optional[Tuple[int, int]] = None
    ) -> "EinTenContraction":
        """Set ``base_tensor`` as diagonal along two of its indices.

        Parameters
        ----------
        base_tensor: :class:`EinTen`
            The tensor to be replaced by its diagonal version. It must represent
            a single base tensor.
        to_identity: bool, optional
            If ``True`` the diagonal tensor is not kept in the contraction and it
            is replaced by the appropriate Kronecker deltas.
        indices: tuple of int, optional
            Pair of tensor axes to contract. If ``None`` and ``base_tensor`` has
            rank 2 the default ``(0, 1)`` is used.
        """

        if not base_tensor.is_base:
            raise ValueError(f"Tensor {base_tensor} is not a base tensor")

        base_prefactor = base_tensor.addends[0].prefactor
        base_tensor_obj = base_tensor.addends[0].indices_and_tensors[0][1]

        rank = len(base_tensor_obj.shape)

        if indices is None:
            if rank != 2:
                raise ValueError("indices must be provided for tensors with rank > 2")
            idx_keep, idx_remove = 0, 1
        else:
            if len(indices) != 2:
                raise ValueError("indices must contain two elements")
            idx_keep, idx_remove = indices
            if not (0 <= idx_keep < rank and 0 <= idx_remove < rank):
                raise ValueError("indices out of range")
            if idx_keep == idx_remove:
                raise ValueError("indices must be different")

        if base_tensor_obj.shape[idx_keep] != base_tensor_obj.shape[idx_remove]:
            raise ValueError(
                f"Cannot contract indices with different dimensions: {base_tensor_obj.shape[idx_keep]} != {base_tensor_obj.shape[idx_remove]}"
            )

        new_shape = list(base_tensor_obj.shape)
        new_shape.pop(idx_remove)
        new_base_tensor = EinTenBaseTensor(
            base_tensor_obj.name, tuple(new_shape), base_tensor_obj.constant
        )

        prefactor = self.prefactor
        indices_and_tensors = []
        new_delta = {}

        for inds, ten in self.indices_and_tensors:
            if ten == base_tensor_obj:
                prefactor /= base_prefactor
                idx_keep_val = inds[idx_keep]
                idx_remove_val = inds[idx_remove]
                new_delta[idx_remove_val] = idx_keep_val
                if not to_identity:
                    new_inds = list(inds)
                    del new_inds[idx_remove]
                    indices_and_tensors.append((new_inds, new_base_tensor))
            else:
                indices_and_tensors.append((list(inds), ten))

        new_delta = {**self.delta, **new_delta}

        new_indices_and_tensors_final = []
        for inds, ten in indices_and_tensors:
            inds = [new_delta.get(i, i) for i in inds]
            new_indices_and_tensors_final.append((inds, ten))

        out_indices = [new_delta.get(i, i) for i in self.out_indices]

        return EinTenContraction(
            tuple(out_indices),
            new_indices_and_tensors_final,
            prefactor=prefactor,
            delta=new_delta,
        )

    def evaluate(self, memory_limit: Optional[str] = None, **kwargs: Any) -> Any:
        # Check if all tensors are constants (no kwargs needed)
        # In this case, use numpy to avoid opt_einsum issues with zero arguments
        all_constants = all(
            ten.constant is not None for _, ten in self.indices_and_tensors
        )
        if utils.oe is None or all_constants:
            return self.evaluate_numpy(**kwargs)
        else:
            return self.evaluate_opt_einsum(memory_limit=memory_limit, **kwargs)

    def evaluate_opt_einsum(self, memory_limit: Optional[str] = None, **kwargs: Any) -> Any:
        # Check if output indices have duplicates (e.g., diagonal like ii or iij)
        if len(set(self.out_indices)) != len(self.out_indices):
            # Handle diagonal output indices similar to evaluate_numpy
            # First, compute with unique indices, then expand to full shape

            # Build index to shape mapping
            idx_shape = {}
            for idx in self.out_indices:
                for indices, ten in self.indices_and_tensors:
                    for i, index in enumerate(indices):
                        if index == idx:
                            if ten.constant is not None:
                                idx_shape[idx] = ten.constant.shape[i]
                            else:
                                idx_shape[idx] = ten.shape[i]

            result_shape = [idx_shape[idx] for idx in self.out_indices]
            unique_indices = list(dict.fromkeys(self.out_indices))  # Preserve order, remove duplicates

            # Build contract expression args with unique output indices
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

            contract_expression_args.append(tuple(unique_indices))
            path = self.get_opt_einsum_path(
                contract_expression_args, constants, memory_limit=memory_limit
            )

            if args:
                result_tmp = self.prefactor * path(*args)
            else:
                result_tmp = self.prefactor * path()

            # Create full result array and assign to diagonal
            result = np.zeros(result_shape, dtype=result_tmp.dtype)
            result_view = np.einsum(result, list(self.out_indices), list(unique_indices))
            result_view[:] = result_tmp

            return result

        # Construct the einsum arguments for non-diagonal case
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
        path = self.get_opt_einsum_path(
            contract_expression_args, constants, memory_limit=memory_limit
        )
        if args:
            return self.prefactor * path(*args)
        else:
            # All tensors are constants - call without arguments
            return self.prefactor * path()

    def get_opt_einsum_path(self, args: List[Any], constants: List[int], memory_limit: Optional[str] = None) -> Any:
        if self.opt_einsum_path is None:
            self.opt_einsum_path = utils.oe.contract_expression(
                *args, constants=constants, memory_limit=memory_limit
            )
        return self.opt_einsum_path

    def evaluate_numpy(self, **kwargs: Any) -> Any:
        """Apply the tensor to the arguments
        Construct the einsum arguments and calling np.einsum
        """

        def _get_tensor_value(ten: EinTenBaseTensor) -> Any:
            """Get tensor value from constant or kwargs."""
            if ten.constant is not None:
                return ten.constant
            return kwargs[ten.name]

        # Construct the einsum arguments
        args = [
            (_get_tensor_value(ten), indices) for indices, ten in self.indices_and_tensors
        ]
        # Flatten the list
        args_flat = [arg for pair in args for arg in pair]

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
                            idx_shape[idx] = _get_tensor_value(ten).shape[i]

            result_shape = []
            for idx in self.out_indices:
                result_shape.append(idx_shape[idx])

            unique_indices = list(set(self.out_indices))

            args_flat.append(unique_indices)
            result_tmp = self.prefactor * np.einsum(
                *args_flat, optimize=self.get_einpath(args_flat)
            )

            # copy the result in the right place
            result = np.zeros(result_shape, dtype=result_tmp.dtype)

            # this create a view and copy the result
            result_view = np.einsum(result, self.out_indices, unique_indices)
            result_view[:] = result_tmp

            return result

        else:
            args_flat.append(self.out_indices)
            return self.prefactor * np.einsum(
                *args_flat, optimize=self.get_einpath(args_flat)
            )

    def get_einpath(self, args: List[Any]) -> Any:
        if self.einsum_path is None:
            try:
                self.einsum_path, _ = np.einsum_path(*args, optimize=True)
            except ValueError:
                raise ValueError(f"Error in einsum_path for {self}")
        return self.einsum_path

    def reduce(
        self,
        old_to_new_indices: Dict[int, int],
        out_indices: Optional[Tuple[int, ...]] = None,
        delta: Optional[Dict[int, int]] = None,
        start_internal_index: Optional[int] = None,
    ) -> "EinTenContraction":
        """
        Perform a reduction of the subscripts.
        old_to_new_indices is a dictionary that maps the old indices to the new ones.
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
            out_indices_final = tuple(
                [local_old_to_new_indices[i] for i in self.out_indices]
            )
        else:
            out_indices_final = out_indices

        # apply the delta to the out_indices
        out_indices_final = tuple(
            new_delta.get(i, i) for i in out_indices_final
        )

        return EinTenContraction(
            out_indices_final,
            new_indices_and_tensors,
            prefactor=self.prefactor,
            delta=new_delta,
        )

    def merge(self, other: "EinTenContraction") -> "EinTenContraction":
        if isinstance(other, EinTenContraction):
            new_indices_and_tensors = (
                self.indices_and_tensors + other.indices_and_tensors
            )
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
            raise ValueError(
                f"Multiplication with type {type(other)} is not supported."
            )

    def __mul__(self, other: Union[float, int]) -> "EinTenContraction":
        return EinTenContraction(
            self.out_indices,
            self.indices_and_tensors,
            delta=self.delta,
            prefactor=self.prefactor * other,
        )

    def pop(self, base_tensor: Any) -> List[Any]:

        if not base_tensor.is_base:
            raise ValueError(f"Tensor {base_tensor} is not a base tensor")

        base_tensor_obj = base_tensor.addends[0].indices_and_tensors[0][1]

        indices_and_tensors = []
        indices = []
        for idx, ten in self.indices_and_tensors:
            if ten == base_tensor_obj:
                indices.append(idx)
            else:
                indices_and_tensors.append((idx, ten))
        self.indices_and_tensors = indices_and_tensors
        return indices
