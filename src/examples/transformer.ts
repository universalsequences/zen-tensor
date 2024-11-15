import { softmax, type TensorGraph, div, sqrt, add, matmul, crossEntropy } from "@/lib";
import { executeEpoch, heInit } from "./core";
import { transpose, reshape } from "@/lib";
import { layerNorm } from "@/lib/layernorm";

const createTransformerModel = (
  g: TensorGraph,
  {
    batchSize,
    seqLength,
    vocabSize,
    embedSize,
    numHeads,
    sequences,
    labels,
  }: {
    batchSize: number;
    seqLength: number;
    vocabSize: number;
    embedSize: number;
    numHeads: number;
    sequences: number[];
    labels: number[];
  },
) => {
  const headDim = embedSize / numHeads;

  // Make input one-hot vectors
  const xOneHot = sequences.flatMap((s) => {
    const oneHot = new Array(vocabSize).fill(0);
    oneHot[s] = 1;
    return oneHot;
  });

  // Initialize weights
  const embeddings = g
    .tensor([embedSize, vocabSize], "embeddings")
    .set(heInit([embedSize, vocabSize]));

  const X = g.tensor([batchSize * seqLength, vocabSize], "sequences").set(xOneHot);

  // Convert labels to one-hot
  const labelTensor = g.tensor([batchSize, vocabSize], "labels").set(
    labels.flatMap((l) => {
      const oneHot = new Array(vocabSize).fill(0);
      oneHot[l] = 1;
      return oneHot;
    }),
  );

  // Attention weights
  const Wq = g.tensor([embedSize, embedSize], "Wq").set(heInit([embedSize, embedSize]));
  const Wk = g.tensor([embedSize, embedSize], "Wk").set(heInit([embedSize, embedSize]));
  const Wv = g.tensor([embedSize, embedSize], "Wv").set(heInit([embedSize, embedSize]));

  // Output projection
  const Wout = g.tensor([embedSize, vocabSize], "Wout").set(heInit([embedSize, vocabSize]));
  const bout = g.tensor([vocabSize], "bout").fill(0);

  // 1. Embed input - all operations in 2D
  const embedded = matmul(X, transpose(embeddings));

  // 2. Self-attention
  // Project to Q, K, V
  const Q = matmul(embedded, Wq);
  const K = matmul(embedded, Wk);
  const V = matmul(embedded, Wv);

  // Need to compute attention scores between each sequence position
  // Reshape to have sequence positions as separate dimensions
  const Q_reshaped = reshape(Q, [batchSize * numHeads, seqLength * headDim]);
  const K_reshaped = reshape(K, [batchSize * numHeads, seqLength * headDim]);
  const V_reshaped = reshape(V, [batchSize * numHeads, seqLength * headDim]);

  // Compute attention scores
  const scores = matmul(Q_reshaped, transpose(K_reshaped));
  const scaled = div(scores, sqrt(headDim));
  const weights = softmax(scaled);

  // Apply attention
  const attended = matmul(weights, V_reshaped);

  // Reshape back
  const output = reshape(attended, [batchSize * seqLength, embedSize]);

  // Get last hidden state (simpler approach)
  const lastHidden = reshape(output, [batchSize, embedSize]);

  const gamma = g.tensor([1, embedSize], "ln_gamma").set(heInit([1, embedSize]));
  const beta = g.tensor([1, embedSize], "ln_beta").set(heInit([1, embedSize], 0.1));

  // Output projection
  //const logits = add(matmul(lastHidden, Wout), bout);
  const normalized = layerNorm(lastHidden, gamma, beta);
  const logits = add(matmul(normalized, Wout), bout);

  const predictions = softmax(logits);
  const entropy = crossEntropy(predictions, labelTensor);

  const loss = entropy;
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [embeddings, gamma, beta, Wq, Wk, Wv, Wout, bout],
    graph: g,
    predictions,
    entropy,
    ignoreZeroLoss: true,
  });
};

export const simpleTransformer = (g: TensorGraph) => {
  return createTransformerModel(g, {
    batchSize: 4,
    seqLength: 3,
    vocabSize: 5,
    embedSize: 32,
    numHeads: 4,
    sequences: [
      0,
      1,
      2, // "the cat sat" -> predict "mat" (3)
      0,
      4,
      2, // "the dog sat" -> predict "mat" (3)
      0,
      1,
      3, // "the cat mat" -> predict "sat" (2)
      0,
      4,
      3, // "the dog mat" -> predict "sat" (2)
    ],
    labels: [3, 3, 2, 2],
  });
};

export const complexTransformer = (g: TensorGraph) => {
  return createTransformerModel(g, {
    batchSize: 8,
    seqLength: 4,
    vocabSize: 12,
    embedSize: 8, // Using simple model params
    numHeads: 2, // Using simple model params
    sequences: [
      0,
      1,
      4,
      7, // "the cat likes food" -> predict "and" (10)
      0,
      2,
      5,
      9, // "the dog chases mice" -> predict "and" (10)
      0,
      3,
      6,
      1, // "the bird watches cat" -> predict "and" (10)
      0,
      1,
      10,
      11, // "the cat and sleep" -> predict "food" (7)
      0,
      2,
      10,
      7, // "the dog and food" -> predict "toys" (8)
      0,
      1,
      5,
      9, // "the cat chases mice" -> predict "and" (10)
      0,
      2,
      6,
      8, // "the dog watches toys" -> predict "and" (10)
      0,
      3,
      4,
      11, // "the bird likes sleep" -> predict "and" (10)
    ],
    labels: [10, 10, 10, 7, 8, 10, 10, 10],
  });
};

export const mediumTransformer_old = (g: TensorGraph) => {
  return createTransformerModel(g, {
    batchSize: 6,
    seqLength: 3,
    vocabSize: 8,
    embedSize: 32, // Using simple model params
    numHeads: 4, // Using simple model params
    sequences: [
      0,
      1,
      3, // "the cat sat" -> predict "mat" (4)
      0,
      2,
      3, // "the dog sat" -> predict "mat" (4)
      0,
      1,
      5, // "the cat ran" -> predict "and" (6)
      0,
      2,
      5, // "the dog ran" -> predict "and" (6)
      0,
      1,
      4, // "the cat mat" -> predict "and" (6)
      0,
      2,
      4, // "the dog mat" -> predict "and" (6)
    ],
    labels: [4, 4, 6, 6, 6, 6],
  });
};

export const mediumTransformer = (g: TensorGraph) => {
  return createTransformerModel(g, {
    batchSize: 8, // Increased to accommodate more balanced examples
    seqLength: 3,
    vocabSize: 8,
    embedSize: 8,
    numHeads: 2,
    sequences: [
      // "mat" predictions (4 examples)
      0,
      1,
      3, // "the cat sat" -> "mat" (4)
      0,
      2,
      3, // "the dog sat" -> "mat" (4)
      0,
      3,
      3, // "the sat sat" -> "mat" (4)
      0,
      5,
      3, // "the ran sat" -> "mat" (4)

      // "and" predictions (4 examples)
      0,
      1,
      5, // "the cat ran" -> "and" (6)
      0,
      2,
      5, // "the dog ran" -> "and" (6)
      0,
      1,
      4, // "the cat mat" -> "and" (6)
      0,
      2,
      4, // "the dog mat" -> "and" (6)
    ],
    labels: [4, 4, 4, 4, 6, 6, 6, 6], // Equal number of each target
  });
};
export const mediumTransformer3 = (g: TensorGraph) => {
  const embedSize = 32; // Define this for clarity
  return createTransformerModel(g, {
    batchSize: 8,
    seqLength: 4,
    vocabSize: 8,
    embedSize: embedSize, // 32
    numHeads: 4,
    sequences: [
      // "mat" predictions (4 examples)
      0,
      1,
      3,
      7, // "the cat sat now" -> "mat"
      0,
      2,
      3,
      7, // "the dog sat now" -> "mat"
      0,
      3,
      3,
      7, // "the sat sat now" -> "mat"
      0,
      5,
      3,
      7, // "the ran sat now" -> "mat"

      // "and" predictions (4 examples)
      0,
      1,
      5,
      7, // "the cat ran now" -> "and"
      0,
      2,
      5,
      7, // "the dog ran now" -> "and"
      0,
      5,
      5,
      7, // "the ran ran now" -> "and"
      0,
      3,
      5,
      7, // "the sat ran now" -> "and"
    ],
    labels: [4, 4, 4, 4, 6, 6, 6, 6],
  });
};

export const sanityCheckTransformer = (g) => {
  const batchSize = 2; // Two examples
  const seqLength = 4; // Sequence length
  const vocabSize = 4; // Vocabulary size
  const embedSize = 8; // Embedding dimension
  const numHeads = 2; // Number of attention heads
  const numClasses = 2; // Two output classes (binary classification)

  // Sequences: [0, 1, 2, 3] → Class 0; [3, 2, 1, 0] → Class 1
  const sequences = [
    0,
    1,
    2,
    3, // Sequence 1
    3,
    2,
    1,
    0, // Sequence 2
  ];

  const labels = [0, 1]; // Class labels

  // Convert labels to one-hot
  const labelTensor = g.tensor([batchSize, numClasses], "labels").set(
    labels.flatMap((label) => {
      const oneHot = new Array(numClasses).fill(0);
      oneHot[label] = 1;
      return oneHot;
    }),
  );

  // Initialize weights
  const embeddings = g
    .tensor([embedSize, vocabSize], "embeddings")
    .set(heInit([embedSize, vocabSize]));

  const Wq = g.tensor([embedSize, embedSize], "Wq").set(heInit([embedSize, embedSize]));
  const Wk = g.tensor([embedSize, embedSize], "Wk").set(heInit([embedSize, embedSize]));
  const Wv = g.tensor([embedSize, embedSize], "Wv").set(heInit([embedSize, embedSize]));
  const Wout = g.tensor([embedSize, numClasses], "Wout").set(heInit([embedSize, numClasses]));
  const bout = g.tensor([numClasses], "bout").fill(0);

  // Create input tensor from sequences
  const xOneHot = sequences.flatMap((s) => {
    const oneHot = new Array(vocabSize).fill(0);
    oneHot[s] = 1;
    return oneHot;
  });

  const X = g.tensor([batchSize * seqLength, vocabSize], "X").set(xOneHot);

  // 1. Embed input
  const embedded = matmul(X, transpose(embeddings));

  // 2. Self-attention
  const Q = matmul(embedded, Wq);
  const K = matmul(embedded, Wk);
  const V = matmul(embedded, Wv);

  const scores = matmul(Q, transpose(K));
  const scaled = div(scores, Math.sqrt(embedSize / numHeads));
  const weights = softmax(scaled);
  const attended = matmul(weights, V);

  // 3. Output projection
  const output = reshape(attended, [batchSize, embedSize]);
  const logits = add(matmul(output, Wout), bout);
  const predictions = softmax(logits);

  // 4. Compute loss
  const entropy = crossEntropy(predictions, labelTensor);
  const loss = entropy;

  // Compile and execute epochs
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [embeddings, Wq, Wk, Wv, Wout, bout],
    graph: g,
    predictions,
    entropy,
    ignoreZeroLoss: true,
  });
};
