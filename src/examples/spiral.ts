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

// Let's create a simple vertical stripes classifier - a problem that should benefit from BatchNorm but is simpler than spirals.
// We'll make alternating vertical stripes of data points and try to classify them:

export const stripesClassifier = (g: TensorGraph) => {
  const inputSize = 2;
  const hiddenSize = 4; // Small hidden layer
  const outputSize = 1;
  const pointsPerClass = 4; // Few points per stripe
  const batchSize = pointsPerClass * 2;

  // Generate simple vertical stripes pattern
  const generateStripes = () => {
    const points: number[] = [];
    const labels: number[] = [];

    // Generate two vertical stripes
    const stripe1X = -0.5; // Left stripe
    const stripe2X = 0.5; // Right stripe

    // Points along each stripe
    for (let i = 0; i < pointsPerClass; i++) {
      const y = -0.75 + (1.5 * i) / (pointsPerClass - 1); // y from -0.75 to 0.75

      // Left stripe (class 0)
      points.push(stripe1X, y);
      labels.push(0);

      // Right stripe (class 1)
      points.push(stripe2X, y);
      labels.push(1);
    }

    return { points, labels };
  };

  const { points, labels } = generateStripes();

  console.log("Sample points:");
  for (let i = 0; i < 4; i++) {
    console.log(
      `Class ${labels[i]}: (${points[i * 2].toFixed(2)}, ${points[i * 2 + 1].toFixed(2)})`,
    );
  }

  const X = g.tensor([batchSize, inputSize], "X").set(points);
  const Y = g.tensor([batchSize, outputSize], "Y").set(labels);

  // Network with BatchNorm
  const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize], 2.0));
  const b1 = g.tensor([hiddenSize], "b1").fill(0);
  const gamma1 = g.tensor([1, hiddenSize], "gamma1").fill(1); // BatchNorm scale
  const beta1 = g.tensor([1, hiddenSize], "beta1").fill(0); // BatchNorm shift

  const W2 = g.tensor([hiddenSize, outputSize], "W2").set(heInit([hiddenSize, outputSize]));
  const b2 = g.tensor([outputSize], "b2").fill(0);

  // Forward pass with BatchNorm
  const hidden = matmul(X, W1);
  const normalized = batchNorm(hidden, gamma1, beta1);
  const activated = leakyRelu(normalized, 0.2);
  const logits = add(matmul(activated, W2), b2);
  const predictions = sigmoid(logits);

  // Log intermediate values to verify BatchNorm
  const loss = g.output(binaryCrossEntropy(predictions, Y));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [W1, gamma1, beta1, W2, b2],
    graph: g,
    predictions,
  });
};

/*
This example:
1. Creates simple vertical stripes of data (-0.5 and +0.5 on x-axis)
2. Should be linearly separable but might have training issues without BatchNorm
3. Has clear logging of pre/post BatchNorm values
4. Uses a minimal network architecture

We should see:
1. Pre-norm values might be large/varied due to weight initialization
2. Post-norm values should be more controlled (roughly mean 0, variance 1)
3. The network should learn to separate the stripes easily

What to verify:
1. Check if pre-norm vs post-norm values show normalization working
2. Verify that predictions improve over time
3. Compare convergence with and without BatchNorm
4. Check if gamma and beta are being updated properly

*/
