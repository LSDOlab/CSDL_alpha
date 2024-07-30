#!/bin/bash

# Run pytest with all possible combinations of the arguments
pytest

pytest --backend jax --build_inline --batched_derivs

pytest --backend jax
pytest --build_inline
pytest --batched_derivs

pytest --backend jax --build_inline
pytest --backend jax --batched_derivs
pytest --build_inline --batched_derivs
