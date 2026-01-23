import multiprocessing as mp
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Sequence

import numpy as np

from eincraft.utils import oe, _quick_sum_prepare, _quick_sum_reduce
from eincraft.symbol import EinTenBaseTensor
from eincraft.term import EinTenContraction


class EinTen:
    """
    Main class representing a tensor or a sum of tensor contractions.

    This class supports symbolic tensor arithmetic and Einstein summation operations.
    """
    slots = ["addends", "ss_to_idx"]

    def __init__(self, name: str, shape: Tuple[int, ...], constant: Optional[Any] = None) -> None:
        """
        Initialize a new EinTen (base tensor).

        Args:
            name: Name of the tensor.
            shape: Shape of the tensor.
            constant: Optional constant value.
        """
        # Create a base tensor
        base_tensor = EinTenBaseTensor(name, shape, constant)
        indices = tuple(range(len(shape)))
        self.addends: List[EinTenContraction] = [
            EinTenContraction(indices, [[indices, base_tensor]])
        ]
        self.ss_to_idx: Dict[str, List[int]] = {}

    @classmethod
    def empty(cls) -> "EinTen":
        """Create an empty EinTen (representing zero)."""
        einten = object.__new__(EinTen)
        einten.addends = []
        einten.ss_to_idx = {}
        return einten

    @property
    def is_base(self) -> bool:
        """Check if this tensor represents a single base tensor."""
        return len(self.addends) == 1 and self.addends[0].is_base

    def get_max_idx(self) -> int:
        """Get the maximum internal index used in this tensor expression."""
        if len(self.addends) == 0:
            return len(self.ss_to_idx)
        return max([a.max_internal_index for a in self.addends])

    @classmethod
    def from_contraction_list(
        cls, addends: List[EinTenContraction], ss_to_idx: Optional[Dict[str, List[int]]] = None
    ) -> "EinTen":
        """Create an EinTen from a list of contractions."""
        einten = object.__new__(EinTen)
        einten.addends = addends
        if ss_to_idx is not None:
            einten.ss_to_idx = ss_to_idx
            # einten.clean_ss_to_idx()
        return einten

    def clean_ss_to_idx(self) -> None:
        """Remove unused subscripts from the ss_to_idx mapping."""
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

    def to_implicit_notation(self) -> None:
        """
        Convert explicit notation to implicit notation if subscripts are unique.
        """
        implicit_ss = None

        # check if its the same for all the addends
        for a in self.addends:

            tmp_ss: Dict[str, int] = {}
            for ss in self.ss_to_idx:
                for idx in a.out_indices:
                    if idx in self.ss_to_idx[ss]:
                        tmp_ss[ss] = tmp_ss.get(ss, 0) + 1

            # Select subscripts that appear exactly once
            tmp_ss_list = [ss for ss, count in tmp_ss.items() if count == 1]
            tmp_ss_list.sort()
            tmp_ss_str = "".join(tmp_ss_list)

            if implicit_ss is None:
                implicit_ss = tmp_ss_str

            if implicit_ss != tmp_ss_str:
                raise ValueError(f"Subscripts are not unique in the output")

        if implicit_ss is None:
            implicit_ss = ""

        self.assign(implicit_ss, self)

    def __eq__(self, other: Any) -> bool:
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

    def __repr__(self) -> str:
        if self.ss_to_idx:
            return "\n".join(
                [a.to_string(ss_to_idx=self.ss_to_idx) for a in self.addends]
            )
        return "\n".join([a.to_string() for a in self.addends])

    def __getattr__(self, subscripts: str) -> "EinTen":
        if subscripts.startswith("_"):
            raise AttributeError(f"Attribute {subscripts} not found.")
        else:
            return self.get_subscripted(tuple(subscripts))

    def __getitem__(self, subscripts: Union[str, Tuple[str, ...]]) -> "EinTen":
        if isinstance(subscripts, str):
            subscripts_tuple = tuple(subscripts)
        elif not isinstance(subscripts, tuple):
            subscripts_tuple = (subscripts,)  # type: ignore
        else:
            subscripts_tuple = subscripts
        return self.get_subscripted(subscripts_tuple)

    def get_subscripted(self, subscripts: Tuple[str, ...]) -> "EinTen":
        """Return a new tensor with the specified subscripts applied."""
        if len(self.ss_to_idx) != 0:
            self.assign(subscripts, self)
            self.ss_to_idx = {}
            return self.get_subscripted(tuple(sorted(subscripts)))

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

    def __setattr__(self, subscripts: str, value: Any) -> None:
        if subscripts.startswith("_") or subscripts in self.slots:
            return super().__setattr__(subscripts, value)
        if isinstance(subscripts, str):
            subscripts_tuple = tuple(subscripts)
        else:
            subscripts_tuple = subscripts # type: ignore
        self.assign(subscripts_tuple, value)

    def __setitem__(self, subscripts: Union[str, Tuple[str, ...]], value: Any) -> None:
        if isinstance(subscripts, str):
            subscripts_tuple = tuple(subscripts)
        elif not isinstance(subscripts, tuple):
            subscripts_tuple = (subscripts,) # type: ignore
        else:
            subscripts_tuple = subscripts
        self.assign(subscripts_tuple, value)

    def assign(self, subscripts: Tuple[str, ...], other: "EinTen") -> None:
        """Assign another tensor expression to this tensor using subscripts."""
        if not isinstance(other, EinTen):
            raise ValueError(f"Assignment with type {type(other)} is not supported.")

        subs_index = {ss: i for i, ss in enumerate(subscripts)}

        max_current_index = max(
            self.get_max_idx(), max(subs_index.values(), default=-1)
        )

        # build the mapping from other's output indices to the target ones
        other_to_self_indices = {}
        for ss in subscripts:
            if ss in other.ss_to_idx:
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
                len(other.ss_to_idx[ss]) == 1
                and other.ss_to_idx[ss][0] == subs_index[ss]
                for ss in subscripts
            )
            and all(
                a.out_indices == new_indices and not a.delta for a in other.addends
            )
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
                    addend.reduce(
                        diag_to_self_indices, out_indices=new_indices, delta=delta
                    )
                )

            for addend in other.addends:
                addends.append(
                    addend.reduce(
                        other_to_self_indices, out_indices=new_indices, delta=delta
                    )
                )

            self.addends += addends
        else:
            addends = []

            for addend in other.addends:
                addends.append(
                    addend.reduce(other_to_self_indices, out_indices=new_indices)
                )

            self.addends = addends

        self.ss_to_idx = {}
        self.simplify()

    @classmethod
    def quick_sum_with_idx(
        cls, eintens: Sequence["EinTen"], idx: Union[str, Iterable[str]]
    ) -> "EinTen":
        """Efficiently sum multiple :class:`EinTen` objects at once.

        Parameters
        ----------
        eintens : Sequence[EinTen]
            Iterable of tensors to sum together.
        idx : Iterable[str]
            Indices for the resulting tensor.  Using
            ``EinTen.quick_sum(eintens, idx)`` is equivalent to::

                res[*idx] = EinTen.quick_sum(eintens)

            but avoids the need for an intermediate tensor.

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

        ss_to_idx: Dict[str, List[int]] = {}
        shift_maps = []

        args = [
            (e, start_other_index, max_idx)
            for e, start_other_index, max_idx in zip(eintens, offsets, max_indices)
        ]

        if len(args) > 1:
            with mp.Pool(min(len(args), mp.cpu_count())) as pool:
                results = pool.map(_quick_sum_prepare, args)
        else:
            results = [_quick_sum_prepare(args[0])]

        for shift_map, start_internal_other_index, part in results:
            shift_maps.append((shift_map, start_internal_other_index))
            for ss, idxs in part.items():
                ss_to_idx[ss] = ss_to_idx.get(ss, []) + idxs

        if isinstance(idx, str):
            idx_tuple = tuple(idx)
        elif not isinstance(idx, tuple):
            idx_tuple = tuple(idx)
        else:
            idx_tuple = idx

        subs_index = {ss: i for i, ss in enumerate(idx_tuple)}

        max_current_index = max(subs_index.values(), default=-1)

        other_to_new_indices = {}
        for ss in idx_tuple:
            for old in ss_to_idx.get(ss, []):
                other_to_new_indices[old] = subs_index[ss]
        for ss, idxs in ss_to_idx.items():
            if ss not in subs_index:
                max_current_index += 1
                for old in idxs:
                    other_to_new_indices[old] = max_current_index

        new_indices = tuple(subs_index[ss] for ss in idx_tuple)

        delta = None
        if len(set(idx_tuple)) != len(idx_tuple):
            delta = {}
            for i, ss in enumerate(idx_tuple):
                first = idx_tuple.index(ss)
                if first != i:
                    delta[i] = first

        reduce_args = [
            (
                einten,
                shift_map,
                start_internal_other_index,
                other_to_new_indices,
                new_indices,
                delta,
            )
            for einten, (shift_map, start_internal_other_index) in zip(
                eintens, shift_maps
            )
        ]

        if len(reduce_args) > 1:
            with mp.Pool(min(len(reduce_args), mp.cpu_count())) as pool:
                results = pool.map(_quick_sum_reduce, reduce_args)
        else:
            results = [_quick_sum_reduce(reduce_args[0])]

        new_addends = [a for sub in results for a in sub]

        return cls.from_contraction_list(new_addends, ss_to_idx={})

    @classmethod
    def quick_sum(cls, eintens: Sequence["EinTen"]) -> "EinTen":
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
        ss_to_idx: Dict[str, List[int]] = {}

        for einten, start_other_index, max_idx in zip(
            eintens, offsets, max_indices
        ):
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
                    ss_to_idx[ss] = (
                        ss_to_idx.get(ss, []) + [start_other_index + idx]
                    )

        return cls.from_contraction_list(addends, ss_to_idx=ss_to_idx)

    def simplify(self) -> None:
        """Simplify the tensor expression by combining equivalent terms."""
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
                if abs(addend.prefactor) > np.finfo(float).eps:
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

    def evaluate(self, memory_limit: Optional[str] = None, **kwargs: Any) -> Any:
        """Apply the tensor to the arguments.

        Constructs the einsum arguments and calls np.einsum (or opt_einsum).
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

    def __mul__(self, other: Union[int, float, complex, "EinTen"]) -> "EinTen":
        if isinstance(other, (int, float, complex)):
            return EinTen.from_contraction_list(
                [a * other for a in self.addends], self.ss_to_idx
            )
        elif isinstance(other, EinTen):

            if len(self.ss_to_idx) == 0 or len(other.ss_to_idx) == 0:
                raise ValueError(
                    f"Cannot multiply not subscripted tensors: {self} and {other}"
                )

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
                    ss_to_idx[ss] = ss_to_idx.get(ss, []) + [
                        start_other_index + idx
                    ]

            return EinTen.from_contraction_list(new_addends, ss_to_idx=ss_to_idx)

        else:
            raise ValueError(f"Multiplication with type {type(other)} is not supported.")

    def __rmul__(self, other: Union[int, float, complex, "EinTen"]) -> "EinTen":
        return self * other

    def __add__(self, other: Union[int, float, complex, "EinTen"]) -> "EinTen":
        if other == 0:
            # Return a copy to ensure immutability of the original if the user expects distinct objects
            einten = EinTen.from_contraction_list([a.copy() for a in self.addends], ss_to_idx=self.ss_to_idx.copy() if self.ss_to_idx else {})
            return einten

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
                    ss_to_idx[ss] = ss_to_idx.get(ss, []) + [
                        start_other_index + idx
                    ]

            return EinTen.from_contraction_list(addends, ss_to_idx=ss_to_idx)

        else:
            raise ValueError(f"Addition with type {type(other)} is not supported.")

    def sum(self, eintens: Sequence["EinTen"]) -> "EinTen":
        """Sum this tensor with a sequence of other tensors."""
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
                start_other_index = max(
                    start_other_index, new_addend.max_internal_index
                )
                new_other_addends.append(new_addend)

            addends += new_other_addends

            ss_to_idx = {ss: idxs.copy() for ss, idxs in self.ss_to_idx.items()}
            for ss, idxs in einten.ss_to_idx.items():
                for idx in idxs:
                    ss_to_idx[ss] = (
                        ss_to_idx.get(ss, []) + [start_other_index + idx]
                    )

        return EinTen.from_contraction_list(addends, ss_to_idx=ss_to_idx)

    def __radd__(self, other: Union[int, float, complex, "EinTen"]) -> "EinTen":
        return self + other

    def __neg__(self) -> "EinTen":
        return -1.0 * self

    def __sub__(self, other: Union[int, float, complex, "EinTen"]) -> "EinTen":
        return self + (-other)

    def __rsub__(self, other: Union[int, float, complex, "EinTen"]) -> "EinTen":
        return other + (-self)

    def __pow__(self, n: int) -> Union[float, "EinTen"]:
        if not isinstance(n, int):
            raise ValueError(f"Power with type {type(n)} is not supported.")
        if n == 0:
            return 1.0
        if n == 1:
            return self
        if n < 0:
            raise ValueError(f"Power < 0 is not supported")
        return self * (self ** (n - 1))

    def substitute(self, tensor: Any, *tensors: Any) -> None:
        """
        Substitute a base tensor with the new tensors
        """
        addends = []
        for addend in self.addends:
            addends += addend.substitute(tensor, *tensors)
        self.addends = addends
        self.simplify()

    def set_as_diagonal(
        self, base_tensor: Any, to_identity: bool = False, indices: Optional[Tuple[int, int]] = None
    ) -> None:
        """Mark ``base_tensor`` as diagonal along the given indices.

        The overall rank of the result is not reduced; instead a Kronecker
        delta is introduced to enforce the equality of the contracted indices.
        """
        self.addends = [
            a.set_as_diagonal(base_tensor, to_identity, indices=indices)
            for a in self.addends
        ]

    def get_addends(self) -> List[Tuple[float, "EinTen"]]:
        result = []
        for addend in self.addends:
            addend_copy = addend.copy()
            addend_copy.prefactor = 1.0
            result.append(
                (
                    addend.prefactor,
                    EinTen.from_contraction_list(
                        [addend_copy], ss_to_idx=self.ss_to_idx
                    ),
                )
            )
        return result

    def pop(self, base_tensor: Any) -> List[List[str]]:
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
