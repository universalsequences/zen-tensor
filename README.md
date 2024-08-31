## zen-tensor

A small tensor library for working with `webgpu`. It utilizes the `zen` approach of lazy compilation.

It's currently imprisoned in a horrible `nextjs` app, but will soon exist as a library.

`zen-tensor` supports simple tensor operations and computes gradients via backpropagation.

You can write mathematical expressions on `tensors` (e.g.`mult(a,b)`) and the compiler will generate and run `webgpu` kernels to compute the expression.

## future

The goal is for this to run extremely simple deep learning models, and to be one-day integrated with `zen-plus` as nodes.

