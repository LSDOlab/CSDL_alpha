# Building More Complex Models

When building complex models in CSDL, it is important to understand how to structure your code to make it easy to read and maintain. This section will cover some best practices for building complex models in CSDL.

## Using functions to organize your code

CSDL variables can be passed through functions just like regular python variables. This can be useful for organizing your code and making it easier to read. For example, you can define a function that takes a variable as an input and returns a new variable as an output. This can help to break up your code into smaller, more manageable pieces.

```python
def my_function(x):
    y = x * 2
    z = y + 1
    return z

recorder = csdl.Recorder()
recorder.start()
x = csdl.Variable((1,))
y = my_function(x)
recorder.stop()
```

## Using classes to organize your code






