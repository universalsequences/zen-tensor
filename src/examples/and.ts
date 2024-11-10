import { TensorGraph, add, matmul, binaryCrossEntropy, sigmoid, leakyRelu } from "@/lib";
import { executeEpoch, heInit } from "./core";

export const andPredictor = (g: TensorGraph) => {
  // Network parameters
  const inputSize = 2;
  const batchSize = 4;
  const outputSize = 1;
  const hiddenSize = 4; // New hidden layer

  // 2. Initialize tensors
  const X = g.tensor([batchSize, inputSize], "X").set([
    0,
    0, // AND(0, 0) = 0
    0,
    1, // AND(0, 1) = 0
    1,
    0, // AND(1, 0) = 0
    1,
    1, // AND(1, 1) = 1
  ]);

  const Y = g.tensor([batchSize, outputSize], "Y").set([
    0, // AND(0, 0) = 0
    0, // AND(0, 1) = 0
    0, // AND(1, 0) = 0
    1, // AND(1, 1) = 1
  ]);

  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize]));
  const b1 = g.tensor([hiddenSize], "b1").fill(0);
  const W2 = g.tensor([hiddenSize, outputSize], "W2").xavierInit();
  const b2 = g.tensor([outputSize], "b2").fill(0);

  // Two-layer neural network
  const hidden = leakyRelu(add(matmul(X, W1), b1));
  const logits = add(matmul(hidden, W2), b2);
  const predictions = sigmoid(logits);

  // Loss function: Binary Cross-Entropy
  const loss = g.output(binaryCrossEntropy(predictions, Y));

  // Compile the computation graph
  g.compile(loss, [4]);

  return executeEpoch({
    tensors: [W1, b1, W2, b2],
    graph: g,
    predictions,
  });
};
