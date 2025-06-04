"""High level tensor API built on top of the contraction primitives."""

import opt_einsum as oe

from .tensor import EinTenBaseTensor
from .contraction import EinTenContraction


class EinTen:
    slots = ["addends", "ss_to_idx"]

    """High level tensor object used to build and evaluate Einstein contractions."""

    def __init__(self, name, shape, constant=None) -> None:
        base_tensor = EinTenBaseTensor(name, shape, constant)
        indices = tuple(range(len(shape)))
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
        return einten

    def clean_ss_to_idx(self):
        internal_indices = set()
        for addend in self.addends:
            for indices, _ in addend.indices_and_tensors:
                internal_indices.update(indices)
        new_ss_to_idx = {}
        for ss, idxs in self.ss_to_idx.items():
            filtered = [idx for idx in idxs if idx in internal_indices]
            if filtered:
                new_ss_to_idx[ss] = filtered
        self.ss_to_idx = new_ss_to_idx

    def to_implicit_notation(self):
        implicit_ss = None
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
                raise ValueError("Subscripts are not unique in the output")
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
                f"  Expected {expected_length} indices but received {actual_length} indices: {subscripts}"
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
        max_current_index = -1
        other_to_self_indices = {}
        for ss in subscripts:
            for idx in other.ss_to_idx[ss]:
                other_to_self_indices[idx] = subscripts.index(ss)
                max_current_index = max(max_current_index, subscripts.index(ss))
        max_current_index = max(max_current_index, self.get_max_idx())
        for ss, idxs in other.ss_to_idx.items():
            if ss not in subscripts:
                max_current_index += 1
                for idx in idxs:
                    other_to_self_indices[idx] = max_current_index
        new_indices = tuple(subscripts.index(ss) for ss in subscripts)
        if len(set(subscripts)) != len(subscripts):
            delta = {}
            for i, ss in enumerate(subscripts):
                if subscripts.index(ss) != i:
                    delta[i] = subscripts.index(ss)
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
            new_indices = tuple(subscripts.index(ss) for ss in subscripts)
            for addend in other.addends:
                addends.append(addend.reduce(other_to_self_indices, out_indices=new_indices))
            self.addends = addends
        self.ss_to_idx = {}
        self.simplify()

    def simplify(self):
        new_addends = []
        for addend in self.addends:
            for new_addend in new_addends:
                if new_addend.equal(addend, check_prefactor=False):
                    new_addend.prefactor += addend.prefactor
                    break
            else:
                new_addends.append(addend)
        self.addends = [a for a in new_addends if a.prefactor != 0.0]
        for addend in self.addends:
            addend.simplify()

    def evaluate(self, **kwargs):
        if len(self.ss_to_idx) != 0:
            self.to_implicit_notation()
        with oe.shared_intermediates():
            return sum([a.evaluate(**kwargs) for a in self.addends])

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return EinTen.from_contraction_list([a * other for a in self.addends], self.ss_to_idx)
        elif isinstance(other, EinTen):
            if len(self.ss_to_idx) == 0 or len(other.ss_to_idx) == 0:
                raise ValueError(f"Cannot multiply not subscripted tensors: {self} and {other}")
            start_other_index = self.get_max_idx() + 1
            other_old_to_new_idx = {}
            other_ss_to_idx = {}
            start_internal_other_index = start_other_index
            for ss, idxs in other.ss_to_idx.items():
                for idx in idxs:
                    other_old_to_new_idx[idx] = start_other_index + idx
                    other_ss_to_idx[ss] = other_old_to_new_idx[idx]
                    start_internal_other_index = max(start_internal_other_index, start_other_index + idx)
            new_other_addends = []
            for a in other.addends:
                new_other_addends.append(
                    a.reduce(other_old_to_new_idx, start_internal_index=start_internal_other_index)
                )
            new_addends = []
            for a1 in self.addends:
                for a2 in new_other_addends:
                    new_addends.append(a1.merge(a2))
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
            return self
        if isinstance(other, EinTen):
            start_other_index = self.get_max_idx() + 1
            start_internal_other_index = start_other_index
            other_old_to_new_idx = {}
            for ss, idxs in other.ss_to_idx.items():
                for idx in idxs:
                    other_old_to_new_idx[idx] = start_other_index + idx
                    start_internal_other_index = max(start_internal_other_index, start_other_index + idx)
            new_other_addends = []
            for a in other.addends:
                new_other_addends.append(
                    a.reduce(other_old_to_new_idx, start_internal_index=start_internal_other_index)
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
                    start_internal_other_index = max(start_internal_other_index, start_other_index + idx)
            new_other_addends = []
            for a in einten.addends:
                new_addend = a.reduce(other_old_to_new_idx, start_internal_index=start_internal_other_index)
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
            raise ValueError("Power < 0 is not supported")
        return self * (self ** (n - 1))

    def substitute(self, tensor, *tensors):
        addends = []
        for addend in self.addends:
            addends += addend.substitute(tensor, *tensors)
        self.addends = addends
        self.simplify()

    def set_as_diagonal(self, base_tensor, to_identity=False):
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
        idx_to_ss = {idx: ss for ss, idxs in self.ss_to_idx.items() for idx in idxs}
        result = []
        for addend in self.addends:
            for idxs in addend.pop(base_tensor):
                result.append([idx_to_ss[idx] for idx in idxs])
        self.clean_ss_to_idx()
        return result

__all__ = ["EinTen", "EinTenBaseTensor", "EinTenContraction"]
