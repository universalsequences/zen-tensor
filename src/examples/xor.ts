import { TensorGraph, add, matmul, relu, binaryCrossEntropy, sigmoid, leakyRelu } from "@/lib";
import { executeEpoch, heInit } from "./core";

// note: this does not work
export const xorPredictor = (g: TensorGraph) => {
  // Network parameters
  const inputSize = 2;
  const batchSize = 4;
  const outputSize = 1;
  const hiddenSize1 = 32; // New hidden layer
  const hiddenSize2 = 16; // New hidden layer

  // 2. Initialize tensors
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

  const Y = g.tensor([batchSize, outputSize], "Y").set([
    0, // XOR(0, 0) = 0
    1, // XOR(0, 1) = 1
    1, // XOR(1, 0) = 1
    0, // XOR(1, 1) = 0
  ]);

  const W1 = g.tensor([inputSize, hiddenSize1], "W1").set(heInit([inputSize, hiddenSize1], 2.0));
  //const W1 = g
  //  .tensor([inputSize, hiddenSize], "W1")
  //  .set([5, 5, -5, -5, 3, 3, -3, -3, -5, 5, -5, 5, -3, 3, -3, 3]);
  const b1 = g.tensor([hiddenSize1], "b1").fill(0.1);
  //const b1 = g.tensor([hiddenSize], "b1").set([-2.5, -7.5, 7.5, 2.5, -1.5, -4.5, 4.5, 1.5]);
  const W2 = g
    .tensor([hiddenSize1, hiddenSize2], "W2")
    .set(heInit([hiddenSize1, hiddenSize2], 2.0));
  const b2 = g.tensor([hiddenSize2], "b2").set(
    Array(hiddenSize2)
      .fill(0)
      .map(() => Math.random() * 2 - 1),
  );
  const W3 = g.tensor([hiddenSize2, outputSize], "W3").set(heInit([hiddenSize2, outputSize], 1.0));
  const b3 = g.tensor([outputSize], "b3").fill(0);

  // Two-layer neural network
  const hidden1 = leakyRelu(add(matmul(X, W1), b1), 0.1);
  const hidden2 = leakyRelu(add(matmul(hidden1, W2), b2), 0.1);
  const logits = add(matmul(hidden2, W3), b3);
  const predictions = sigmoid(logits);

  // Loss function: Binary Cross-Entropy
  const loss = g.output(binaryCrossEntropy(predictions, Y));

  // Compile the computation graph
  g.compile(loss, [4]);

  return executeEpoch({
    tensors: [W1, b1, W2, b2, W3, b3],
    graph: g,
    predictions,
  });
};
