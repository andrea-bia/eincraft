"""Contraction logic for EinCraft tensors."""

import numpy as np
import opt_einsum as oe

from .tensor import EinTenBaseTensor
from .index import IndexMap


def _idx_to_str(idx):
    if idx > 9:
        return chr(0x208D) + "".join(chr(0x2080 + int(i)) for i in str(idx)) + chr(0x208E)
    return chr(0x2080 + idx)


class EinTenContraction:
    """Represents a single contraction addend."""

    def __init__(self, out_indices, indices_and_tensors, prefactor=1.0, delta=None) -> None:
        self.indices_and_tensors = indices_and_tensors
        self.out_indices = out_indices
        self.prefactor = prefactor
        self.delta = IndexMap(delta)
        self.einsum_path = None
        self.opt_einsum_path = None

        if self.indices_and_tensors:
            self.max_internal_index = max([max(indices) for indices, _ in self.indices_and_tensors])
        else:
            self.max_internal_index = -1

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
        """Return a canonical map representation for equality checks."""
        if (
            (self._map is None and not with_count)
            or (self._detailed_map is None and with_count)
        ):
            idxs = set([idx for indices, _ in self.indices_and_tensors for idx in indices])
            tensor_count = {}
            contractions_map = {s: [] for s in idxs}

            for s in self.out_indices:
                positions = tuple(i for i, val in enumerate(self.out_indices) if val == s)
                if with_count:
                    contractions_map[s].append(("", positions, 0))
                else:
                    contractions_map[s].append(("", positions))

            for indices, tensor in self.indices_and_tensors:
                tensor_count[tensor.name] = tensor_count.get(tensor.name, 0) + 1
                position_by_index = {}
                for pos, idx in enumerate(indices):
                    position_by_index.setdefault(idx, []).append(pos)

                for idx, pos_list in position_by_index.items():
                    if with_count:
                        contractions_map[idx].append((tensor.name, tuple(pos_list), tensor_count[tensor.name]))
                    else:
                        contractions_map[idx].append((tensor.name, tuple(pos_list)))

            equivalent_deltas = {}
            for idx in self.delta:
                root_idx = self.delta.find(idx)
                equivalent_deltas.setdefault(root_idx, []).append(idx)

            for delta_count, (k, eqv_ks) in enumerate(equivalent_deltas.items()):
                if with_count:
                    for v in eqv_ks:
                        contractions_map[k].append(("_ec_delta", (v,), delta_count))
                else:
                    for v in eqv_ks:
                        contractions_map[k].append(("_ec_delta", (v,)))

            contractions_map = [tuple(sorted(contractions_map[k])) for k in contractions_map]

            if with_count:
                self._detailed_map = tuple(sorted(contractions_map))
            else:
                self._map = tuple(sorted(contractions_map))

        return self._detailed_map if with_count else self._map

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
            [f"{ten.name}" + "".join([idx_to_str(i) for i in indices]) for indices, ten in self.indices_and_tensors]
        )
        if self.delta:
            result += "".join(f"δ{idx_to_str(k)}{idx_to_str(v)}" for k, v in self.delta.items())
        result += ")"
        result += "".join([idx_to_str(i) for i in self.out_indices])

        idx_to_ss = {idxs[0]: ss for ss, idxs in ss_to_idx.items()}
        result += " {" + ", ".join(f"{idx_to_idx_for_print[idx]}➞{ss}" for idx, ss in idx_to_ss.items()) + "}"
        return result

    def to_string(self, ss_to_idx=None):
        if ss_to_idx is not None:
            return self.to_string_with_subscripts(ss_to_idx)

        result = f"{self.prefactor:+5f} ("
        result += "".join(
            [f"{ten.name}" + "".join([_idx_to_str(i) for i in indices]) for indices, ten in self.indices_and_tensors]
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

    def substitute(self, old_tensor, *new_tensors):
        if not old_tensor.is_base:
            raise ValueError(f"Tensor {old_tensor} is not a base tensor")

        old_prefactor = old_tensor.addends[0].prefactor
        old_tensor = old_tensor.addends[0].indices_and_tensors[0][1]

        new_addends = [EinTenContraction(self.out_indices, [], prefactor=self.prefactor, delta=self.delta.copy())]
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

        new_delta = IndexMap()
        for k, v in self.delta.items():
            new_delta.add(
                old_to_new_internal_indices.get(k, k),
                old_to_new_internal_indices.get(v, v),
            )
        self.delta = new_delta

    def set_as_diagonal(self, base_tensor, to_identity=False):
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
        new_delta = IndexMap()

        for indices, ten in self.indices_and_tensors:
            if ten == base_tensor:
                prefactor /= base_prefactor
                new_delta.add(max(indices), min(indices))
                if not to_identity:
                    indices_and_tensors.append(([indices[0]], new_base_tensor))
            else:
                indices_and_tensors.append((indices, ten))

        new_delta = new_delta | self.delta
        new_indices_and_tensors = []
        for indices, ten in indices_and_tensors:
            indices = [new_delta.get(i, i) for i in indices]
            new_indices_and_tensors.append((indices, ten))
        indices_and_tensors = new_indices_and_tensors

        out_indices = [new_delta.get(i, i) for i in self.out_indices]
        return EinTenContraction(out_indices, indices_and_tensors, prefactor=prefactor, delta=new_delta)

    def evaluate(self, **kwargs):
        return self.evaluate_numpy(**kwargs)

    def evaluate_opt_einsum(self, **kwargs):
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
        contract_expression_args.append(self.index)
        return self.prefactor * self.get_opt_einsum_path(contract_expression_args, constants)(*args)

    def get_opt_einsum_path(self, args, constants):
        if self.opt_einsum_path is None:
            self.opt_einsum_path = oe.contract_expression(*args, constants=constants)
        return self.opt_einsum_path

    def evaluate_numpy(self, **kwargs):
        args = [(kwargs[ten.name], indices) for indices, ten in self.indices_and_tensors]
        args = [arg for pair in args for arg in pair]
        if len(set(self.out_indices)) != len(self.out_indices):
            idx_shape = {}
            for idx in self.out_indices:
                for indices, ten in self.indices_and_tensors:
                    for i, index in enumerate(indices):
                        if index == idx:
                            idx_shape[idx] = kwargs[ten.name].shape[i]
            result_shape = [idx_shape[idx] for idx in self.out_indices]
            unique_indices = list(set(self.out_indices))
            args.append(unique_indices)
            result_tmp = self.prefactor * np.einsum(*args, optimize=self.get_einpath(args))
            result = np.zeros(result_shape, dtype=result_tmp.dtype)
            result_view = np.einsum(result, self.out_indices, unique_indices)
            result_view[:] = result_tmp
            return result
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
        local_old_to_new_indices = old_to_new_indices.copy()
        max_internal_index = start_internal_index
        if max_internal_index is None:
            max_internal_index = self.max_internal_index
        new_delta = IndexMap()
        for k, v in self.delta.items():
            if k not in local_old_to_new_indices:
                max_internal_index += 1
                local_old_to_new_indices[k] = max_internal_index
            if v not in local_old_to_new_indices:
                max_internal_index += 1
                local_old_to_new_indices[v] = max_internal_index
            new_delta.add(local_old_to_new_indices[k], local_old_to_new_indices[v])

        if delta is not None:
            for k, v in delta.items():
                if k in new_delta:
                    new_delta.add(v, new_delta[k])
                else:
                    new_delta.add(k, v)

        new_indices_and_tensors = []
        for indices, tensor in self.indices_and_tensors:
            new_indices = []
            for i in indices:
                if i not in local_old_to_new_indices:
                    max_internal_index += 1
                    local_old_to_new_indices[i] = max_internal_index
                if local_old_to_new_indices[i] in new_delta:
                    new_indices.append(new_delta[local_old_to_new_indices[i]])
                else:
                    new_indices.append(local_old_to_new_indices[i])
            new_indices_and_tensors.append((new_indices, tensor))

        if out_indices is None:
            out_indices = tuple([local_old_to_new_indices[i] for i in self.out_indices])
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
            delta=self.delta.copy(),
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

__all__ = ["EinTenContraction", "_idx_to_str", "IndexMap"]
