import { TensorGraph, add, matmul, relu, meanSquaredError, sigmoid, leakyRelu, tanh } from "@/lib";
import { executeEpoch, heInit } from "./core";
import { batchNorm } from "@/lib/batchnorm";

export const sineLearner = (g: TensorGraph) => {
  const inputSize = 1;
  const hiddenSize = 16;
  const outputSize = 1;
  const batchSize = 16;

  // Generate x values between -π and π
  const xValues: number[] = [];
  const yValues: number[] = [];

  for (let i = 0; i < batchSize; i++) {
    const x = -Math.PI + (2 * Math.PI * i) / (batchSize - 1);
    // Scale inputs to be smaller
    xValues.push(x / Math.PI); // Now between -1 and 1
    yValues.push(Math.sin(x)); // Already between -1 and 1
  }

  const X = g.tensor([batchSize, inputSize], "X").set(xValues);
  const Y = g.tensor([batchSize, outputSize], "Y").set(yValues);

  // Smaller initial weights
  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize], 0.5));
  const gamma1 = g.tensor([1, hiddenSize], "gamma1").fill(1);
  const beta1 = g.tensor([1, hiddenSize], "beta1").fill(0);

  const W2 = g.tensor([hiddenSize, outputSize], "W2").set(heInit([hiddenSize, outputSize], 0.1));
  const b2 = g.tensor([outputSize], "b2").fill(0);

  const hidden = matmul(X, W1);
  const normalized = batchNorm(hidden, gamma1, beta1);
  const activated = tanh(leakyRelu(normalized, 0.1));
  const output = add(matmul(activated, W2), b2);
  const predictions = output;

  const loss = g.output(meanSquaredError(predictions, Y));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W1, gamma1, beta1, W2, b2],
    graph: g,
    predictions,
  });
};
