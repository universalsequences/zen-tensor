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

export const circleClassifier = (g: TensorGraph) => {
  // Simple dataset: classify points based on y coordinate
  const batchSize = 8;
  const inputSize = 2;
  const hiddenSize = 4;
  const outputSize = 1;

  // Generate points: positive y = class 1, negative y = class 0
  const points: number[] = [
    0,
    0.5, // Point in top half (y > 0)
    0,
    -0.5, // Point in bottom half (y < 0)
    0.5,
    0.5, // Point in top half
    0.5,
    -0.5, // Point in bottom half
    -0.5,
    0.5, // Point in top half
    -0.5,
    -0.5, // Point in bottom half
    0.25,
    0.5, // Point in top half
    0.25,
    -0.5, // Point in bottom half
  ];

  const labels = [
    1, // y > 0
    0, // y < 0
    1,
    0,
    1,
    0,
    1,
    0,
  ];

  const X = g.tensor([batchSize, inputSize], "X").set(points);
  const Y = g.tensor([batchSize, outputSize], "Y").set(labels);

  // Single layer network - this problem is linearly separable
  const W = g.tensor([inputSize, outputSize], "W").set([
    0.0, // x weight - initialize to 0
    0.1, // y weight - small positive weight for y coordinate
  ]);
  const b = g.tensor([outputSize], "b").fill(0);

  // Simple linear model with no hidden layer
  const logits = add(matmul(X, W), b);

  // Let's log the intermediate values
  const predictions = sigmoid(logits);

  const loss = g.output(binaryCrossEntropy(predictions, Y));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W, b],
    graph: g,
    predictions,
  });
};
