# Debugging

CSDL provides a few tools to help you debug your code. This section will cover some of the most common debugging techniques in CSDL.

## Using the Recorder

The `Recorder` class provides a few functionalities to help you debug your code. The most useful is inline evaluation, which allows you to execute your code inline and access the values of your variables. This is activated by passing `inline=True` to the `Recorder` constructor. 

Similarly, you can pass `debug=True` to the `Recorder` constructor, which causes each variable to store a trace to the file and line where it was created. This can be useful for debugging, as it allows you to see where each variable was created in your code. Traces can be seen by calling `print_trace()` on a variable.

Two methods of visualizing your model are also provided. The `visualize_graph()` method of the `Recorder` class will create a visualization of the graph, which can be useful for understanding the structure of your model. The `visualize_adjacency_matrix()` method of the `Recorder` class will create a visualization of the adjacency matrix, which can be useful for understanding the flow of data through your model.

```{warning}
Graph visualization can be slow for large models.
```
