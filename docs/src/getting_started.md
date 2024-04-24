# Getting started
This page provides instructions for installing CSDL 
and running a minimal example.

## Installation

`csdl_alpha` is tested on Linux, and may not work on other operating systems.

### Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
$ pip install git+https://github.com/LSDOlab/csdl_alpha.git
```

### Installation instructions for developers
To install `csdl_alpha`, first clone the repository and install using pip. On the terminal or command line, run
```sh
$ git clone https://github.com/LSDOlab/csdl_alpha.git
$ pip install -e ./csdl_alpha
```

## Testing

To run all tests for the CSDL frontend, navigate to the CSDL directory and, from terminal, run
```sh
$ pytest
```


## Writing CSDL Code

Writing CSDL code is similar to writing regular python code. However, CSDL code is compiled to a graph representation by the [`Recorder`](api_references/recorder.md) class, that can be optimized and executed efficiently. This means that CSDL code is not executed immediately, but rather compiled to a graph that is executed by the CSDL backend. However, the CSDL frontend provides a way to execute CSDL code inline, which can be useful for debugging and testing. Inline values can be accessed by the `value` attribute of the variable.

```{warning}
No backends currently exist for CSDL. 
```

### Basic Example

The following is an example of a simple CSDL code snippet that adds two variables `x` and `y` and stores the result in `z`. The code is executed inline by passing an argument to the [`Recorder`](api_references/recorder.md) class. All CSDL code should be enclosed within the `start()` and `stop()` methods of the [`Recorder`](api_references/recorder.md) class.

```python
import csdl_alpha as csdl
recorder = csdl.Recorder(inline=True)
recorder.start()
x = csdl.Variable(value=1)
y = csdl.Variable(value=2)
z = x+y
recorder.stop()
```
