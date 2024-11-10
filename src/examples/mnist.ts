import { TensorGraph, add, matmul, relu, binaryCrossEntropy, sigmoid, leakyRelu } from "@/lib";
import { executeEpoch, heInit } from "./core";

export const digitClassifier = (g: TensorGraph) => {
  // Network parameters
  const inputSize = 784; // 28x28 pixels
  const hiddenSize = 128;
  const outputSize = 1; // Binary classification: is it a 7 or not
  const batchSize = 8; // Small batch for testing

  // Sample data: Let's create some very simple "7" and "not 7" patterns
  // This is a highly simplified version - you'd want real MNIST data
  const createSeven = () => {
    const pixels = new Array(784).fill(0);
    // Horizontal line at top
    for (let i = 0; i < 10; i++) pixels[i] = 1;
    // Diagonal line
    for (let i = 0; i < 20; i++) pixels[10 + i * 28 + i] = 1;
    return pixels;
  };

  const createNotSeven = () => {
    const pixels = new Array(784).fill(0);
    // Vertical line (like "1")
    for (let i = 0; i < 28; i++) pixels[i * 28 + 14] = 1;
    return pixels;
  };

  // Create training data
  const X = g.tensor([batchSize, inputSize], "X").set([
    ...createSeven(), // 7
    ...createNotSeven(), // not 7 (1)
    ...createSeven(), // 7
    ...createNotSeven(), // not 7 (1)
    ...createSeven(), // 7
    ...createNotSeven(), // not 7 (1)
    ...createSeven(), // 7
    ...createNotSeven(), // not 7 (1)
  ]);

  const Y = g.tensor([batchSize, outputSize], "Y").set([
    1, // 7
    0, // not 7
    1, // 7
    0, // not 7
    1, // 7
    0, // not 7
    1, // 7
    0, // not 7
  ]);

  // Two-layer network
  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize], 2.0));
  const b1 = g.tensor([hiddenSize], "b1").fill(0.1);

  const W2 = g.tensor([hiddenSize, outputSize], "W2").set(heInit([hiddenSize, outputSize]));
  const b2 = g.tensor([outputSize], "b2").fill(0);

  const hidden = leakyRelu(add(matmul(X, W1), b1), 0.2);
  const logits = add(matmul(hidden, W2), b2);
  const predictions = sigmoid(logits);

  const loss = g.output(binaryCrossEntropy(predictions, Y));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W1, b1, W2, b2],
    graph: g,
    predictions,
  });
};
