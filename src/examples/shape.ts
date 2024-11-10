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

export const shapeClassifier = (g: TensorGraph) => {
  const gridSize = 6; // Smaller grid
  const inputSize = gridSize * gridSize;
  const hiddenSize = 16;
  const outputSize = 1;
  const batchSize = 8;

  // Helper to create very distinct patterns
  const createTriangle = () => {
    const grid = new Array(gridSize * gridSize).fill(0);
    // Simple right triangle in corner
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x <= y; x++) {
        grid[y * gridSize + x] = 1;
      }
    }
    return grid;
  };

  const createBox = () => {
    const grid = new Array(gridSize * gridSize).fill(0);
    // Simple box pattern
    for (let y = 1; y < gridSize - 1; y++) {
      for (let x = 1; x < gridSize - 1; x++) {
        grid[y * gridSize + x] = 1;
      }
    }
    return grid;
  };

  // Create training data with very distinct patterns
  const patterns: number[] = [];
  const labels: number[] = [];

  // Add alternating triangles and boxes
  for (let i = 0; i < 4; i++) {
    const triangle = createTriangle();
    const box = createBox();

    // No random noise this time
    patterns.push(...triangle);
    patterns.push(...box);

    labels.push(1); // Triangle
    labels.push(0); // Box
  }

  const X = g.tensor([batchSize, inputSize], "X").set(patterns);
  const Y = g.tensor([batchSize, outputSize], "Y").set(labels);

  // Similar initialization to MNIST example that worked
  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize], 2.0));
  const b1 = g.tensor([hiddenSize], "b1").fill(0.1);

  const W2 = g.tensor([hiddenSize, outputSize], "W2").set(heInit([hiddenSize, outputSize]));
  const b2 = g.tensor([outputSize], "b2").fill(0);

  const hidden = leakyRelu(add(matmul(X, W1), b1), 0.2);
  const logits = add(matmul(hidden, W2), b2);
  const predictions = sigmoid(logits);

  console.log("Initial predictions:", predictions.data);
  console.log("Logits:", logits.data);

  const loss = g.output(binaryCrossEntropy(predictions, Y));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W1, b1, W2, b2],
    graph: g,
    predictions,
    learningRate: 0.1, // Same as MNIST
  });
};
