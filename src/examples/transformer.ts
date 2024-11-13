import { softmax, type TensorGraph, div, sqrt, add, matmul, crossEntropy } from "@/lib";
import { executeEpoch, heInit } from "./core";
import { transpose, reshape } from "@/lib";

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

  const X = g.tensor([batchSize * seqLength, vocabSize], "X").set(xOneHot);

  // Convert labels to one-hot
  const labelTensor = g.tensor([batchSize, vocabSize], "Y").set(
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

  // Output projection
  const logits = add(matmul(lastHidden, Wout), bout);
  const predictions = softmax(logits);
  const entropy = crossEntropy(predictions, labelTensor);

  const loss = g.output(entropy);
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [embeddings, Wq, Wk, Wv, Wout, bout],
    graph: g,
    predictions,
    entropy,
  });
};

export const simpleTransformer = (g: TensorGraph) => {
  return createTransformerModel(g, {
    batchSize: 4,
    seqLength: 3,
    vocabSize: 5,
    embedSize: 8,
    numHeads: 2,
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

export const mediumTransformer = (g: TensorGraph) => {
  return createTransformerModel(g, {
    batchSize: 6,
    seqLength: 3,
    vocabSize: 8,
    embedSize: 8, // Using simple model params
    numHeads: 2, // Using simple model params
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
