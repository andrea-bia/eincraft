import numpy as np
from eincraft import EinTen


def test_sum():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij + A2.ij
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2) + np.einsum("ij->ji", a2))

    O.ji = A2.ij + A2.ji
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2) + np.einsum("ji->ji", a2))

    O.ji = A2.ij + 2.0 * B2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2),
        np.einsum("ij->ji", a2) + 2.0 * np.einsum("ji->ji", b2),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih + A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk->kji", a2, b2, a3),
    )


def test_self_sum():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij + A2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) + np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ij = O.ij + A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) + np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ji = O.ji + A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) + np.einsum("ij->ji", a2),
    )

    O.kji = A2.ij * B2.jk
    O.kij = O.kij + A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk->kji", a2, b2) + np.einsum("ijk->kij", a3),
    )

    O.kji = A2.ij * B2.jh * A3.kji * A2.ih
    O.kij = O.kij + A2.ij * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2) + np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij = O.kij + A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij += A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.ji = A2.ij - A2.ji
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2) - np.einsum("ji->ji", a2))

    O.ji = A2.ij - 2.0 * B2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2),
        np.einsum("ij->ji", a2) - 2.0 * np.einsum("ji->ji", b2),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih - A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        - 2.0 * np.einsum("ij,id,idk->kji", a2, b2, a3),
    )


def test_diff():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij - A2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) - np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ij = O.ij - A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) - np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ji = O.ji - A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) - np.einsum("ij->ji", a2),
    )

    O.kji = A2.ij * B2.jk
    O.kij = O.kij - A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk->kji", a2, b2) - np.einsum("ijk->kij", a3),
    )

    O.kji = A2.ij * B2.jh * A3.kji * A2.ih
    O.kij = O.kij - A2.ij * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2) - np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij = O.kij - A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        - 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij -= A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        - 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )


def test_sum_method_matches_chained_addition_and_does_not_mutate_self():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    C2 = EinTen("C2", (3, 3))

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    c2 = np.random.rand(3, 3)

    base = A2.ij
    before = base.evaluate(A2=a2).copy()

    batched = base.sum([B2.ij, C2.ij])

    out_batched = EinTen.empty()
    out_batched.ij = batched

    np.testing.assert_allclose(
        out_batched.evaluate(A2=a2, B2=b2, C2=c2),
        a2 + b2 + c2,
    )

    np.testing.assert_allclose(base.evaluate(A2=a2), before)
