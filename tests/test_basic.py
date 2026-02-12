import numpy as np
from eincraft import EinTen, disable_opt_einsum


def test_basic_ops():
    A2 = EinTen("A2", (3, 3))
    
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)

    O.ji = A2.ij
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2))

    O.ji = 2.0 * A2.ij
    assert np.allclose(O.evaluate(A2=a2), 2.0 * np.einsum("ij->ji", a2))

    O.ji = np.sqrt(2.0) * A2.ij
    assert np.allclose(O.evaluate(A2=a2), np.sqrt(2.0) * np.einsum("ij->ji", a2))

    O.i = A2.ii
    assert np.allclose(O.evaluate(A2=a2), np.diag(a2))


def test_mul_unity():
    A1 = EinTen("A1", (3,))
    a1 = np.random.rand(3)

    result = 1.0
    result *= A1.i
    assert np.allclose(result.evaluate(A1=a1), a1) 


def test_neg():
    A2 = EinTen("A2", (3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)

    O.ji = -A2.ij
    assert np.allclose(O.evaluate(A2=a2), -np.einsum("ij->ji", a2))

    O.ji = -A2.ij
    O.ji -= -A2.ij
    assert np.allclose(O.evaluate(A2=a2), -np.einsum("ij->ji", a2) + np.einsum("ij->ji", a2))


def test_getattr_on_subscripted():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O = (A2.ij * B2.jk).ik
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ik", a2, b2))

    O = (A2.ij * B2.jk).ki
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ki", a2, b2))

    O = (A2.ij * B2.jk).ijk
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ijk", a2, b2))

    O = A2.ij * B2.jk
    O.ik = O.ik + A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk", a2, b2) + np.einsum("ijk->ik", a3),
    )

    O = A2.ij * B2.jk
    O.ki = O.ik + A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk->ki", a2, b2) + np.einsum("ijk->ki", a3),
    )

    O = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O += A2.ij * 2.0 * B2.id * A3.idk * A2.jj
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk,jj", a2, b2, a3, a2),
    )


def test_evaluate_numpy_with_constant():
    """Test that evaluate_numpy handles base tensor constants like opt_einsum."""
    a2 = np.random.rand(3, 3)
    
    # Create tensor with constant value
    A2 = EinTen("A2", (3, 3), constant=a2)
    
    # This should work - using constant value without passing in kwargs
    # Currently fails with KeyError because evaluate_numpy ignores ten.constant
    try:
        result = A2.ij.evaluate()
        assert np.allclose(result, a2)
    except KeyError as e:
        # This demonstrates the bug - numpy backend doesn't handle constants
        assert "A2" in str(e)
        raise AssertionError(
            "evaluate_numpy() does not handle base tensor constants. "
            "It should use ten.constant when available, matching opt_einsum behavior."
        ) from e
