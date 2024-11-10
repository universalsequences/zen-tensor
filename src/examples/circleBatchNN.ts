import {
  TensorGraph,
  add,
  matmul,
  mult,
  relu,
  binaryCrossEntropy,
  sigmoid,
  leakyRelu,
} from "@/lib";
import { executeEpoch, heInit } from "./core";
import { batchNorm } from "@/lib/batchnorm";

export const scaleClassifier = (g: TensorGraph) => {
  // Small dataset
  const batchSize = 6;
  const inputSize = 2;
  const hiddenSize = 4;
  const outputSize = 1;

  // Create data where x1 is normal scale but x2 is very large scale
  const points = [
    0.5,
    1000, // Class 1 (large second feature)
    0.3,
    1200, // Class 1
    0.4,
    800, // Class 1
    -0.5,
    -900, // Class 0 (negative large second feature)
    -0.3,
    -1100, // Class 0
    -0.4,
    -950, // Class 0
  ];

  const labels = [
    1, // Positive x2
    1,
    1,
    0, // Negative x2
    0,
    0,
  ];

  const X = g.tensor([batchSize, inputSize], "X").set(points);
  const Y = g.tensor([batchSize, outputSize], "Y").set(labels);

  // Network with BatchNorm
  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize], 1.0));
  const gamma1 = g.tensor([1, hiddenSize], "gamma1").fill(1);
  const beta1 = g.tensor([1, hiddenSize], "beta1").fill(0);

  const W2 = g.tensor([hiddenSize, outputSize], "W2").set(heInit([hiddenSize, outputSize]));
  const b2 = g.tensor([outputSize], "b2").fill(0);

  const hidden = matmul(X, W1);
  const normalizedHidden = batchNorm(hidden, gamma1, beta1);
  const activatedHidden = leakyRelu(normalizedHidden, 0.2);
  const logits = add(matmul(activatedHidden, W2), b2);
  const predictions = sigmoid(logits);

  const loss = g.output(binaryCrossEntropy(predictions, Y));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W1, W2, b2],
    graph: g,
    predictions,
  });
};
