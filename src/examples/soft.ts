import { mult, sub, log, softmax, reshape, crossEntropy } from "@/lib";
import {
  TensorGraph,
  add,
  div,
  matmul,
  relu,
  meanSquaredError,
  sigmoid,
  leakyRelu,
  tanh,
} from "@/lib";
import { executeEpoch, heInit } from "./core";
import { batchNorm } from "@/lib/batchnorm";

export const testTwoSentenceUncertaintyMatmul2D = (g: TensorGraph) => {
  const numClasses = 5; // Vocabulary size
  const batchSize = 2; // Two sentences
  const seqLength = 3; // Each sentence has 3 tokens

  // Input token sequences
  const sequencesData = [
    0,
    1,
    2, // "The cat sat"
    0,
    1,
    3, // "The cat ate"
  ];
  const sequences = g.tensor([batchSize, seqLength], "sequences").set(sequencesData);

  // Raw labels
  const rawLabels = [
    1,
    2,
    4, // "cat", "sat", "<end>"
    1,
    3,
    4, // "cat", "ate", "<end>"
  ];
  const labels = g.tensor([batchSize, seqLength], "labels").set(rawLabels);

  // One-hot encode sequences
  const oneHotSequences = g.tensor([batchSize * seqLength, numClasses], "oneHotSequences").set(
    sequencesData.flatMap((id) => {
      const oneHot = new Array(numClasses).fill(0);
      oneHot[id] = 1;
      return oneHot;
    }),
  );

  // One-hot encode labels
  const oneHotLabels = g.tensor([batchSize * seqLength, numClasses], "oneHotLabels").set(
    rawLabels.flatMap((l) => {
      const oneHot = new Array(numClasses).fill(0);
      oneHot[l] = 1;
      return oneHot;
    }),
  );

  // Define model weights and biases
  const Wout = g.tensor([numClasses, numClasses], "Wout").set(heInit([numClasses, numClasses]));
  const bout = g.tensor([numClasses], "bout").fill(0);

  // Compute logits
  const logits = add(matmul(oneHotSequences, Wout), bout); // Shape: [6, 5]

  // Compute softmax
  const softmaxOutput = softmax(logits); // Shape: [6, 5]

  // Compute log of softmax
  const loss = crossEntropy(softmaxOutput, oneHotLabels);
  // Compile and run
  g.compile(g.output(loss), [batchSize]); // Loss is scalar
  return executeEpoch({
    tensors: [Wout, bout],
    graph: g,
    predictions: softmaxOutput,
    entropy: loss,
    ignoreZeroLoss: false,
  });
};

export const testSoftmaxCrossEntropyMinimal = (g: TensorGraph) => {
  const numClasses = 3; // Small vocabulary size
  const batchSize = 2; // Two examples

  // Define input logits (raw scores before softmax)
  const logitsData = [
    1,
    2,
    3, // First example
    3,
    2,
    1, // Second example
  ];
  const logits = g.tensor([batchSize, numClasses], "logits").set(logitsData);

  // Define one-hot encoded labels
  const labelsData = [
    0,
    0,
    1, // First example: Correct class is 3rd
    1,
    0,
    0, // Second example: Correct class is 1st
  ];
  const labels = g.tensor([batchSize, numClasses], "labels").set(labelsData);

  // Compute softmax
  const softmaxOutput = softmax(logits);

  // Compute cross-entropy loss
  const loss = crossEntropy(softmaxOutput, labels);

  // Compile and execute
  g.compile(g.output(loss), [batchSize]); // Loss is scalar
  return executeEpoch({
    tensors: [],
    graph: g,
    predictions: softmaxOutput,
    entropy: loss,
  });
};

export const testLearnableParameters = (g: TensorGraph) => {
  const numClasses = 3; // Vocabulary size (3 tokens: "A", "B", "C")
  const batchSize = 2; // Two sequences
  const seqLength = 2; // Two tokens per sequence

  // Input sequences (encoded as token indices)
  const sequencesData = [
    0,
    1, // "A B"
    1,
    2, // "B C"
  ];
  const sequences = g.tensor([batchSize, seqLength], "sequences").set(sequencesData);

  // Target labels: the next token in each sequence
  const rawLabels = [
    1,
    2, // "B", "C"
    2,
    0, // "C", "A"
  ];
  const labels = g.tensor([batchSize, seqLength], "labels").set(rawLabels);

  // One-hot encode sequences
  const oneHotSequences = g.tensor([batchSize * seqLength, numClasses], "oneHotSequences").set(
    sequencesData.flatMap((id) => {
      const oneHot = new Array(numClasses).fill(0);
      oneHot[id] = 1;
      return oneHot;
    }),
  );

  // One-hot encode labels
  const oneHotLabels = g.tensor([batchSize * seqLength, numClasses], "oneHotLabels").set(
    rawLabels.flatMap((l) => {
      const oneHot = new Array(numClasses).fill(0);
      oneHot[l] = 1;
      return oneHot;
    }),
  );

  // Define model parameters (learnable tensors)
  const Wout = g.tensor([numClasses, numClasses], "Wout").set(heInit([numClasses, numClasses]));
  const bout = g.tensor([numClasses], "bout").fill(0);

  // Forward pass
  const logits = add(matmul(oneHotSequences, Wout), bout); // Shape: [batchSize * seqLength, numClasses]
  const softmaxOutput = softmax(logits); // Shape: [batchSize * seqLength, numClasses]
  const loss = crossEntropy(softmaxOutput, oneHotLabels); // Scalar loss

  // Compile the graph
  g.compile(loss, [batchSize]);

  // Train for multiple epochs
  return executeEpoch({
    tensors: [Wout, bout],
    graph: g,
    predictions: softmaxOutput,
    entropy: loss,
    ignoreZeroLoss: false,
  });
};

// actually works!
export const testSoftmaxBinaryClassification = (g: TensorGraph) => {
  const numClasses = 2; // Binary classification
  const inputSize = 2; // Two input features (2D points)
  const batchSize = 4; // Four training examples

  // Define the dataset: points and their labels
  const inputsData = [
    0.0,
    0.0, // Class 0
    0.0,
    1.0, // Class 1
    1.0,
    0.0, // Class 0
    1.0,
    1.0, // Class 1
  ]; // Shape: [4, 2]
  const labelsData = [
    1,
    0, // Class 0 (one-hot)
    0,
    1, // Class 1 (one-hot)
    1,
    0, // Class 0 (one-hot)
    0,
    1, // Class 1 (one-hot)
  ]; // Shape: [4, 2]

  // Input and label tensors
  const inputs = g.tensor([batchSize, inputSize], "inputs").set(inputsData);
  const labels = g.tensor([batchSize, numClasses], "labels").set(labelsData);

  // Model: Single layer with softmax
  const W_out = g.tensor([inputSize, numClasses], "W_out").set(heInit([inputSize, numClasses]));
  const b_out = g.tensor([numClasses], "b_out").fill(0);

  // Forward pass: Compute logits and probabilities
  const logits = add(matmul(inputs, W_out), b_out); // Shape: [4, 2]
  const probabilities = softmax(logits); // Shape: [4, 2]

  // Compute cross-entropy loss
  const loss = crossEntropy(probabilities, labels);

  // Compile and train
  g.compile(loss, [batchSize]); // Loss is scalar
  return executeEpoch({
    tensors: [W_out, b_out],
    graph: g,
    predictions: probabilities,
    entropy: loss,
    ignoreZeroLoss: false,
  });
};

export const testNonLinearClassification = (g: TensorGraph) => {
  const numFeatures = 2; // 2D input space
  const numHidden = 4; // Hidden layer units
  const numClasses = 3; // 3 classes
  const numSamples = 99; // Dataset size

  // Synthetic dataset
  const data = [
    [0.1, 0.3], // Cluster 1
    [0.5, 0.7], // Cluster 2
    [0.9, 0.2], // Cluster 3
  ].flatMap(([x, y], i) =>
    Array.from({ length: numSamples / 3 }, () => [
      x + Math.random() * 0.1,
      y + Math.random() * 0.1,
      i, // Label
    ]),
  );
  const inputs = data.map(([x, y]) => [x, y]).flat();
  const labels = data.map(([, , label]) => label);

  const inputTensor = g.tensor([numSamples, numFeatures], "inputs").set(inputs);
  const labelTensor = g.tensor([numSamples, numClasses], "labels").set(
    labels.flatMap((l) => {
      const oneHot = Array(numClasses).fill(0);
      oneHot[l] = 1;
      return oneHot;
    }),
  );

  // Learnable weights
  const W1 = g.tensor([numFeatures, numHidden], "W1").set(heInit([numFeatures, numHidden]));
  const b1 = g.tensor([numHidden], "b1").fill(0);
  const W2 = g.tensor([numHidden, numClasses], "W2").set(heInit([numHidden, numClasses]));
  const b2 = g.tensor([numClasses], "b2").fill(0);

  // Forward pass
  const hidden = relu(add(matmul(inputTensor, W1), b1));
  const logits = add(matmul(hidden, W2), b2);
  const predictions = softmax(logits);

  // Loss
  const loss = crossEntropy(predictions, labelTensor);

  // Compile and execute
  g.compile(loss, [numSamples]);
  return executeEpoch({
    tensors: [W1, b1, W2, b2],
    graph: g,
    predictions,
    entropy: loss,
    ignoreZeroLoss: false,
  });
};

export const testClusteredData = (g: TensorGraph) => {
  const numSamples = 33; // Divisible by numClasses
  const numFeatures = 2; // Number of features per input
  const numHidden = 8; // Number of hidden units
  const numClasses = 3; // Number of output classes

  // Generate data
  const data = [
    [0.1, 0.3], // Cluster 1
    [0.5, 0.7], // Cluster 2
    [0.9, 0.2], // Cluster 3
  ].flatMap(([x, y], i) =>
    Array.from({ length: numSamples / numClasses }, () => [
      x + Math.random() * 0.1,
      y + Math.random() * 0.1,
      i, // Label
    ]),
  );

  // Split inputs and labels
  const inputs = data.map(([x, y]) => [x, y]); // Shape: [99, 2]
  const labels = data.map(([, , label]) => label); // Shape: [99]

  // Convert inputs and labels to tensors
  const inputTensor = g.tensor([numSamples, numFeatures], "inputs").set(inputs.flat()); // Shape: [99, 2]
  const labelTensor = g.tensor([numSamples, numClasses], "labels").set(
    labels.flatMap((l) => {
      const oneHot = Array(numClasses).fill(0);
      oneHot[l] = 1; // One-hot encode labels
      return oneHot;
    }),
  ); // Shape: [99, 3]

  // Model parameters
  const W1 = g.tensor([numFeatures, numHidden], "W1").set(heInit([numFeatures, numHidden])); // Shape: [2, 8]
  const b1 = g.tensor([numHidden], "b1").fill(0); // Shape: [8]
  const W2 = g.tensor([numHidden, numClasses], "W2").set(heInit([numHidden, numClasses])); // Shape: [8, 3]
  const b2 = g.tensor([numClasses], "b2").fill(0); // Shape: [3]

  // Forward pass
  const hidden = leakyRelu(add(matmul(inputTensor, W1), b1), 0.01); // Shape: [99, 8]
  const logits = add(matmul(hidden, W2), b2); // Shape: [99, 3]
  const predictions = softmax(logits); // Shape: [99, 3]

  // Compute loss
  const loss = crossEntropy(predictions, labelTensor); // Scalar

  // Compile and train
  g.compile(loss, [numSamples]);
  return executeEpoch({
    tensors: [W1, b1, W2, b2],
    graph: g,
    predictions,
    entropy: loss,
    ignoreZeroLoss: false,
  });
};

export const testSubDivSoftmax = (g: TensorGraph) => {
  const batchSize = 3; // Number of samples
  const numClasses = 4; // Number of output classes

  // Input logits
  const logits = g
    .tensor([batchSize, numClasses], "logits")
    .set([2.0, 1.0, 0.1, -1.0, 0.5, -0.5, 1.5, -2.0, -1.0, 2.0, 0.0, -0.5]);

  // Labels (one-hot encoded)
  const labels = g
    .tensor([batchSize, numClasses], "labels")
    .set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);

  // Apply transformations to test `sub` and `div`
  const logitsSub = sub(logits, 1.0); // Subtract a scalar
  const logitsDiv = div(logitsSub, 2.0); // Divide by a scalar

  // Apply softmax to the transformed logits
  const predictions = softmax(logitsDiv);

  // Compute loss
  const loss = crossEntropy(predictions, labels);

  // Compile and Execute
  g.compile(loss, [batchSize]);
  return executeEpoch({
    tensors: [logits],
    graph: g,
    predictions,
    entropy: loss,
    ignoreZeroLoss: false,
  });
};
