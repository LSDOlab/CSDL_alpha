## FOR WORKSHOP

Sparse matrices are currently not natively supported as of April 2024. They are planned to be integrated slowly in the upcoming weeks. Please use the ```csdl.SparseMatrix``` class for now (it is just a dense variable).

```python
import numpy as np

sparse_matrix = csdl.SparseMatrix(value = <np array>)
```

# Support for sparse matrices

CSDL supports sparse matrix variables and a limited set of operations on them.
In this page, we will go over all the features of CSDL sparse variables.

## 1. Initializing a sparse matrix variable

There are mainly two ways in which a sparse variable cam be initialized in CSDL.
The first one uses a scipy `coo`, `csr`, or `csc` matrix to intialize the
sparse variable as shown below.

```python
import numpy as np
from scipy.sparse import coo_array
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
scipy_mtx=coo_array((data, (row, col)), shape=(4, 4))

import csdl
s1 = csdl.SparseVariable(scipy_mtx=scipy_mtx, name='s1')
```

Another way to initialize a sparse variable is by manually
entering the nonzero `value` vector along with any two of
the vectors in `(rows, cols, ind_ptr)`, where `rows` and `cols`
are the row and column indices corresponding to the `value` vector.
`ind_ptr` is the vector containing the list of indices of the `value`
vector where each column (for csc) or row (for csr) starts.
The user also needs to explicity specify the `shape` of the matrix
in this case. 
Note that `value` can be input as a scalar in which case the
CSDL automatically creates a vector having size inferred from the
other inputs and fill it with scalar value provided.
An example for this initialization is shown below.

```python
import numpy as np
rows  = np.array([0, 3, 1, 0])
cols  = np.array([0, 3, 1, 2])
value = np.array([4, 5, 7, 9])

import csdl
# coo
s2 = csdl.SparseVariable(
    rows=rows, 
    cols=cols, 
    value=value, 
    shape=(4,4), 
    name='s2'
    )
# coo with scalar value
s3 = csdl.SparseVariable(
    rows=rows, 
    cols=cols, 
    value=1., 
    shape=(4,4), 
    name='s3'
    )

# csr
rows  = np.array([0, 0, 1, 3])
cols  = np.array([0, 2, 1, 3])
value = np.array([4, 9, 7, 5])
ind_ptr = [0, 2, 3, 3, 4]
s4 = csdl.SparseVariable(
    ind_ptr=ind_ptr, 
    cols=cols, 
    value=value, 
    shape=(4,4), 
    name='s4'
    )

# csc
rows  = np.array([0, 1, 0, 3])
cols  = np.array([0, 1, 2, 3])
value = np.array([4, 7, 9, 5])
ind_ptr = [0, 1, 2, 3, 4]
s5 = csdl.SparseVariable(
    ind_ptr=ind_ptr, 
    rows=rows, 
    value=value, 
    shape=(4,4), 
    name='s5'
    )
```

```{warning}
In situations involving multiple runs of a CSDL model/graph, e.g.,
during optimization or uncertainty quantification,
CSDL expects the sparsity structures of the sparse
variables to remain the same during multiple runs of the model/graph.
Note particularly that in the subsequent runs of the model after initialization,
CSDL only updates the `value` vector assuming that the ordering
of nonzeros in the initialized `value` vector remains unchanged,
i.e., all the vectors `rows`, `cols`, and `ind_ptr` are assumed to be constants
in the subsequent runs.
```


## 2. Utilities for sparse matrix variables

CSDL provides two utilities for conversion between sparse and dense variables.
Given an input sparse variable, the `sparse2dense` function returns a normal 
dense csdl variable. 
The `dense2sparse` function returns a csdl sparse variable corresponding to
an input dense variable.

```python
import numpy as np
from scipy.sparse import coo_array
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
scipy_mtx=coo_array((data, (row, col)), shape=(4, 4))

import csdl
s1 = csdl.SparseVariable(scipy_mtx=scipy_mtx, name='s1')
# sparse to dense
d1 = csdl.sparse2dense(s1)

d2 = csdl.Variable(value=np.zeros((4,4)), name='d2')
# dense to sparse 
s2 = csdl.dense2sparse(d2)
```


## 3. Operations supported

A limited number of operations are supported for sparse variables.



### 3.1 Basic math

Basic math operations such as addition, subtraction, multiplication,
division, and exponentiation are supported.
The complete list of basic math operations and API examples are shown below.


```{list-table} Basic math operations
:header-rows: 1
:name: basic-math

* - Operation
  - API example
  - Exceptions
* - add
  - `csdl.add(a,b)`
  - a and b must be both sparse?
* - subtract
  - `csdl.subtract(a,b)`
  - a and b must be both sparse?
* - multiply
  - `csdl.multiply(a,b)`
  - 
* - divide
  - `csdl.divide(a,b)`
  - `b` cannot be a sparse variable.
* - power
  - `csdl.power(a,b)`
  - can b be sparse?
* - square root
  - `csdl.sqrt(a)`
  - 
* - exponential
  - `csdl.power(np.e,a)`
  -  ?
* - log(1+a)
  - `csdl.log1p(a)`
  -  ?
```

```{note}
Unless otherwise specified,
`a`,`b` can be normal csdl variables, sparse variables, constants, or scalars.
```
```{warning}
`log 0` is undefined, and therefore, log of a sparse matrix is undefined.
```



### 3.2 Linear algebra

The following list of linear algebra operations are supported for sparse variables.

```{list-table} Linear algebra operations
:header-rows: 1
:name: linear-algebra

* - Operation
  - API example
  - Exceptions
* - blockmat
  - `csdl.blockmat([[A,B],[C,D]])`
  - 
* - matvec
  - `csdl.matvec(A,b)`
  - 
* - matmat
  - `csdl.matmat(A,B)`
  -
* - linear_solve
  - `csdl.linear_solve(A,b)`
  - 
* - transpose
  - `csdl.transpose(A)`
  -
* - norm
  - `csdl.norm(A, type=2)`
  -

```



### 3.3 Trigonometric operations

The following list shows the set of sparse trigonometric operations supported by CSDL.

```{list-table} Trigonometric operations
:header-rows: 1
:name: trig-ops

* - Operation
  - API example
  - Exceptions
* - sin
  - `csdl.sin(a)`
  - 
* - cos
  - `csdl.cos(a)`
  - ?
* - tan
  - `csdl.tan(a)`
  - 
* - asin
  - `csdl.asin(a)`
  - 
* - acos
  - `csdl.acos(a)`
  - ?
* - atan
  - `csdl.atan(a)`
  -
* - sinh
  - `csdl.sinh(a)`
  -
* - cosh
  - `csdl.cosh(a)`
  - ?
* - tanh
  - `csdl.tanh(a)`
  -
```




### 3.4 Aggregation operations

The list of sparse aggregation operations supported by csdl are shown below.

```{list-table} Aggregation operations
:header-rows: 1
:name: aggregate-ops

* - Operation
  - API example
  - Exceptions
* - average
  - `csdl.average(a)`
  - 
* - sum
  - `csdl.sum(a)`
  -
* - max
  - `csdl.max(a)`
  -
* - min
  - `csdl.min(a)`
  - 

```

### 3.4 Other operations

The list of additional sparse operations supported by csdl are shown below.

```{list-table} Other operations
:header-rows: 1
:name: other-ops

* - Operation
  - API example
  - Exceptions
* - reshape
  - `csdl.reshape(a, newshape)`
  - 
* - get
  - `a[slice]`, `a[index_list]`
  -
* - set
  - `a.set(slice, b)`, \
    `a.set(index_list, b)`
  -
```
