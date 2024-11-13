import { TensorGraph, add, matmul, relu, binaryCrossEntropy, sigmoid, leakyRelu } from "@/lib";
import { executeEpoch, heInit } from "./core";
import { batchNorm } from "@/lib/batchnorm";

// learning rate -> 0.1 actually works!
export const xorPredictor = (g: TensorGraph) => {
  const inputSize = 2;
  const batchSize = 4;
  const outputSize = 1;
  const hiddenSize = 8;

  const X = g.tensor([batchSize, inputSize], "X").set([
    0,
    0, // XOR(0, 0) = 0
    0,
    1, // XOR(0, 1) = 1
    1,
    0, // XOR(1, 0) = 1
    1,
    1, // XOR(1, 1) = 0
  ]);

  const Y = g.tensor([batchSize, outputSize], "Y").set([0, 1, 1, 0]);

  // Custom initialization to force different patterns
  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize], 3)); /*[
    // First input connections
    ...Array(hiddenSize / 2).fill(3), // Strong positive for first half
    ...Array(hiddenSize / 2).fill(-3), // Strong negative for second half
    // Second input connections
    ...Array(hiddenSize / 2).fill(-3), // Opposite pattern for second input
    ...Array(hiddenSize / 2).fill(3),
  ]);
  */

  const gamma1 = g.tensor([1, hiddenSize], "gamma1").fill(1);
  const beta1 = g.tensor([1, hiddenSize], "beta1").set(
    Array(hiddenSize)
      .fill(0)
      .map((_, i) => (i % 2 === 0 ? 0.5 : -0.5)),
  );

  const W2 = g.tensor([hiddenSize, outputSize], "W2").set(
    Array(hiddenSize)
      .fill(0)
      .map((_, i) => (i < hiddenSize / 2 ? 1.0 : -1.0)),
  );
  const b2 = g.tensor([outputSize], "b2").fill(0);

  const hidden = matmul(X, W1);
  const normalized = batchNorm(hidden, gamma1, beta1);
  const activated = leakyRelu(normalized, 0.1);
  const logits = add(matmul(activated, W2), b2);
  const predictions = sigmoid(logits);
  const entropy = binaryCrossEntropy(predictions, Y);

  const loss = g.output(entropy);
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W1, gamma1, beta1, W2, b2],
    graph: g,
    predictions,
    entropy,
  });
};
