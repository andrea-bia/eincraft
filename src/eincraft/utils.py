from typing import Any, Dict, List, Tuple, Union

try:
    import opt_einsum as oe
except Exception:  # pragma: no cover - optional dependency
    oe = None


def disable_opt_einsum() -> None:
    """
    Disable the opt_einsum library if it is installed.
    This is useful for testing purposes or when you want to avoid using opt_einsum.
    """
    global oe
    oe = None


def find_root(key: int, mapping: Dict[int, int]) -> int:
    """
    Follow the mapping chain from key and return its final representative.
    If a cycle is detected, choose a canonical representative from the cycle
    (here, the minimum key in the cycle) and update the mapping for all members.

    Args:
        key: The key to find the root for.
        mapping: The dictionary containing the mapping.

    Returns:
        The root representative for the key.
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


def _idx_to_str(idx: int) -> str:
    """
    Convert an integer index to a subscript string representation.

    Args:
        idx: The integer index.

    Returns:
        The string representation (using unicode subscripts).
    """
    if idx > 9:
        return chr(0x208D) + "".join(chr(0x2080 + int(i)) for i in str(idx)) + chr(0x208E)
    return chr(0x2080 + idx)


def _quick_sum_prepare(args: Tuple[Any, int, int]) -> Tuple[Dict[int, int], int, Dict[str, List[int]]]:
    """
    Prepare shift maps for multiprocessing in :func:`EinTen.quick_sum`.

    Args:
        args: A tuple containing (einten, start_other_index, max_idx).

    Returns:
        A tuple containing (shift_map, next_start_index, partial_ss_to_idx).
    """
    einten, start_other_index, max_idx = args
    shift_map = {}
    partial_ss_to_idx: Dict[str, List[int]] = {}
    for ss, idxs in einten.ss_to_idx.items():
        for i_idx in idxs:
            global_idx = start_other_index + i_idx
            shift_map[i_idx] = global_idx
            partial_ss_to_idx.setdefault(ss, []).append(global_idx)
    return shift_map, start_other_index + max_idx, partial_ss_to_idx


def _quick_sum_reduce(
    args: Tuple[Any, Dict[int, int], int, Dict[int, int], Tuple[int, ...], Union[Dict[int, int], None]]
) -> List[Any]:
    """
    Reduce addends for multiprocessing in :func:`EinTen.quick_sum`.

    Args:
        args: A tuple containing:
            - einten: The tensor object.
            - shift_map: Mapping for shifting indices.
            - start_internal_other_index: The starting internal index for the other tensor.
            - other_to_new_indices: Mapping from other indices to new indices.
            - new_indices: The target output indices.
            - delta: Optional delta dictionary.

    Returns:
        A list of reduced addends.
    """
    einten, shift_map, start_internal_other_index, other_to_new_indices, new_indices, delta = args
    combined_map = {old: other_to_new_indices[shift_map[old]] for old in shift_map}
    sub_addends = []
    for a in einten.addends:
        sub_addends.append(
            a.reduce(
                combined_map,
                out_indices=new_indices,
                delta=delta,
                start_internal_index=start_internal_other_index,
            )
        )
    return sub_addends
