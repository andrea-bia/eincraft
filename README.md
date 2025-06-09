# eincraft

eincraft is an experimental Python library designed to simplify and automate einsum operations in a more intuitive and user-friendly way. In its current alpha state, the library allows you to write einsum operations in a more straightforward and readable format, such as `C.ik = A.ij * B.jk`, rather than using the traditional einsum notation.

## Installation

To install eincraft, you can use pip:

```bash
pip install .
```

If you want to enable support for the optional `opt_einsum` backend, install it with:

```bash
pip install .[opt_einsum]
```


## Usage

Here's an example of how to use eincraft:

```python
import numpy as np
import eincraft as ec

# Create tensor objects
A = ec.EinTen('A')
B = ec.EinTen('B')
C = ec.EinTen('C')
Z = ec.EinTen('Z')


# Write einsum operation
Z.kji = 2.0 * A.ij * B.jk * C.ijk + 4.0 * A.ij * 2.0 * B.id * C.idk
Z.ijk += 0.25 * A.ij * B.jk * C.kdd

# Define the numpy arrays
a = np.random.rand(3, 3)
b = np.random.rand(3, 3)
c = np.random.rand(3, 3, 3)

# Evaluate the einsum operations
z = Z.evaluate(A=a, B=b, C=c)

# Let's check the result
np.allclose(z, 2.0 * np.einsum('ij,jk,ijk->kji', a, b, c) + 4.0 * np.einsum('ij,id,idk->ijk', a, 2.0 * b, c) + 0.25 * np.einsum('ij,jk,kdd->ijk', a, b, c))

```

## Running the tests

To run the tests, you can use the following command:

```bash
pytest
```
