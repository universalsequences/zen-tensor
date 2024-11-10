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

export const shapeNoiseClassifier = (g: TensorGraph) => {
  const gridSize = 6;
  const inputSize = gridSize * gridSize;
  const hiddenSize = 16;
  const outputSize = 1;
  const batchSize = 8;
  const noiseLevel = 0.1; // Start with 10% noise

  // Helper to create patterns with controlled noise
  const addNoise = (value: number) => {
    // Add noise but clamp between 0 and 1 to keep reasonable bounds
    return Math.max(0, Math.min(1, value + (Math.random() * 2 - 1) * noiseLevel));
  };

  const createTriangle = () => {
    const grid = new Array(gridSize * gridSize).fill(0);
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x <= y; x++) {
        // Add noise to the 1s (pattern)
        grid[y * gridSize + x] = addNoise(1);
      }
    }
    // Add noise to the 0s (background)
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] === 0) {
        grid[i] = addNoise(0);
      }
    }
    return grid;
  };

  const createBox = () => {
    const grid = new Array(gridSize * gridSize).fill(0);
    for (let y = 1; y < gridSize - 1; y++) {
      for (let x = 1; x < gridSize - 1; x++) {
        // Add noise to the 1s (pattern)
        grid[y * gridSize + x] = addNoise(1);
      }
    }
    // Add noise to the 0s (background)
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] === 0) {
        grid[i] = addNoise(0);
      }
    }
    return grid;
  };

  // Create training data
  const patterns: number[] = [];
  const labels: number[] = [];

  // Add alternating triangles and boxes
  for (let i = 0; i < 4; i++) {
    const triangle = createTriangle();
    const box = createBox();

    patterns.push(...triangle);
    patterns.push(...box);

    labels.push(1); // Triangle
    labels.push(0); // Box
  }

  // Log a sample pattern to see the noise effect
  console.log("Sample triangle pattern:");
  for (let y = 0; y < gridSize; y++) {
    console.log(
      patterns
        .slice(y * gridSize, (y + 1) * gridSize)
        .map((x) => x.toFixed(2))
        .join(" "),
    );
  }

  const X = g.tensor([batchSize, inputSize], "X").set(patterns);
  const Y = g.tensor([batchSize, outputSize], "Y").set(labels);

  // Same architecture as before
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
