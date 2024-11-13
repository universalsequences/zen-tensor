import { softmax, TensorGraph, div, sqrt, add, matmul, crossEntropy } from "@/lib";
import { executeEpoch, heInit } from "./core";
import { transpose, reshape } from "@/lib";

export const simpleTransformer = (g: TensorGraph) => {
  // Hyperparameters
  const batchSize = 4;
  const seqLength = 3;
  const vocabSize = 5;
  const embedSize = 8;
  const numHeads = 2;
  const headDim = embedSize / numHeads; // = 4

  // Input sequences
  const sequences = [
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
  ];
  const labels = [3, 3, 2, 2];

  // Make input one-hot vectors
  const xOneHot = sequences
    .map((s) => {
      const oneHot = new Array(vocabSize).fill(0);
      oneHot[s] = 1;
      return oneHot;
    })
    .flat();

  // Initialize weights
  const embeddings = g
    .tensor([embedSize, vocabSize], "embeddings")
    .set(heInit([embedSize, vocabSize]));

  const X = g.tensor([batchSize * seqLength, vocabSize], "X").set(xOneHot);

  // Convert labels to one-hot
  const labelTensor = g.tensor([batchSize, vocabSize], "Y").set(
    labels
      .map((l) => {
        const oneHot = new Array(vocabSize).fill(0);
        oneHot[l] = 1;
        return oneHot;
      })
      .flat(),
  );

  // Attention weights
  const Wq = g.tensor([embedSize, embedSize], "Wq").set(heInit([embedSize, embedSize]));
  const Wk = g.tensor([embedSize, embedSize], "Wk").set(heInit([embedSize, embedSize]));
  const Wv = g.tensor([embedSize, embedSize], "Wv").set(heInit([embedSize, embedSize]));

  // Output projection
  const Wout = g.tensor([embedSize, vocabSize], "Wout").set(heInit([embedSize, vocabSize]));
  const bout = g.tensor([vocabSize], "bout").fill(0);

  // 1. Embed input - all operations in 2D
  // [batchSize * seqLength, vocabSize] × [vocabSize, embedSize]
  const embedded = matmul(X, transpose(embeddings)); // [batchSize * seqLength, embedSize]

  // 2. Self-attention
  // Project to Q, K, V
  const Q = matmul(embedded, Wq); // [batchSize * seqLength, embedSize]
  const K = matmul(embedded, Wk); // [batchSize * seqLength, embedSize]
  const V = matmul(embedded, Wv); // [batchSize * seqLength, embedSize]

  // Need to compute attention scores between each sequence position
  // Reshape to have sequence positions as separate dimensions
  const Q_reshaped = reshape(Q, [batchSize * numHeads, seqLength * headDim]);
  const K_reshaped = reshape(K, [batchSize * numHeads, seqLength * headDim]);
  const V_reshaped = reshape(V, [batchSize * numHeads, seqLength * headDim]);

  // Compute attention scores
  const scores = matmul(Q_reshaped, transpose(K_reshaped)); // [batchSize * numHeads, batchSize * numHeads]
  const scaled = div(scores, sqrt(headDim));
  const weights = softmax(scaled);

  // Apply attention
  const attended = matmul(weights, V_reshaped); // [batchSize * numHeads, seqLength * headDim]

  // Reshape back
  const output = reshape(attended, [batchSize * seqLength, embedSize]);

  // Get last hidden state (simpler approach)
  const lastHidden = reshape(output, [batchSize, embedSize]);

  // Output projection
  const logits = add(matmul(lastHidden, Wout), bout);
  const predictions = softmax(logits);

  const loss = g.output(crossEntropy(predictions, labelTensor));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [embeddings, Wq, Wk, Wv, Wout, bout],
    graph: g,
    predictions,
  });
};

export const complexTransformer = (g: TensorGraph) => {
  // Bigger model
  const batchSize = 8;
  const seqLength = 4;
  const vocabSize = 12;
  const embedSize = 16;
  const numHeads = 4;
  const headDim = embedSize / numHeads; // = 4

  // Vocabulary:
  // 0: "the"
  // 1: "cat"
  // 2: "dog"
  // 3: "bird"
  // 4: "likes"
  // 5: "chases"
  // 6: "watches"
  // 7: "food"
  // 8: "toys"
  // 9: "mice"
  // 10: "and"
  // 11: "sleep"

  // More complex sequences with different patterns
  const sequences = [
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
  ];

  // More varied predictions
  const labels = [10, 10, 10, 7, 8, 10, 10, 10];

  // Make input one-hot vectors
  const xOneHot = sequences
    .map((s) => {
      const oneHot = new Array(vocabSize).fill(0);
      oneHot[s] = 1;
      return oneHot;
    })
    .flat();

  // Initialize weights
  const embeddings = g
    .tensor([embedSize, vocabSize], "embeddings")
    .set(heInit([embedSize, vocabSize]));

  const X = g.tensor([batchSize * seqLength, vocabSize], "X").set(xOneHot);

  // Convert labels to one-hot
  const labelTensor = g.tensor([batchSize, vocabSize], "Y").set(
    labels
      .map((l) => {
        const oneHot = new Array(vocabSize).fill(0);
        oneHot[l] = 1;
        return oneHot;
      })
      .flat(),
  );

  // Attention weights
  const Wq = g.tensor([embedSize, embedSize], "Wq").set(heInit([embedSize, embedSize]));
  const Wk = g.tensor([embedSize, embedSize], "Wk").set(heInit([embedSize, embedSize]));
  const Wv = g.tensor([embedSize, embedSize], "Wv").set(heInit([embedSize, embedSize]));

  // Output projection
  const Wout = g.tensor([embedSize, vocabSize], "Wout").set(heInit([embedSize, vocabSize]));
  const bout = g.tensor([vocabSize], "bout").fill(0);

  // 1. Embed input
  const embedded = matmul(X, transpose(embeddings));

  // 2. Self-attention
  const Q = matmul(embedded, Wq);
  const K = matmul(embedded, Wk);
  const V = matmul(embedded, Wv);

  const Q_reshaped = reshape(Q, [batchSize * numHeads, seqLength * headDim]);
  const K_reshaped = reshape(K, [batchSize * numHeads, seqLength * headDim]);
  const V_reshaped = reshape(V, [batchSize * numHeads, seqLength * headDim]);

  const scores = matmul(Q_reshaped, transpose(K_reshaped));
  const scaled = div(scores, add(sqrt(headDim), 0.0001));
  const weights = softmax(scaled);

  const attended = matmul(weights, V_reshaped);
  const output = reshape(attended, [batchSize * seqLength, embedSize]);
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
    entropy: entropy,
  });
};

export const mediumTransformer = (g: TensorGraph) => {
  // Hyperparameters - between simple and complex
  const batchSize = 6; // simple:4, complex:8
  const seqLength = 3; // Keep at 3 like simple version
  const vocabSize = 8; // simple:5, complex:12
  const embedSize = 12; // simple:8, complex:16
  const numHeads = 3; // simple:2, complex:4
  const headDim = embedSize / numHeads; // = 4

  // Vocabulary:
  // 0: "the"
  // 1: "cat"
  // 2: "dog"
  // 3: "sat"
  // 4: "mat"
  // 5: "ran"
  // 6: "and"
  // 7: "toy"

  // Input sequences - keeping patterns clear but more variety
  const sequences = [
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
  ];

  const labels = [4, 4, 6, 6, 6, 6];

  // Make input one-hot vectors
  const xOneHot = sequences
    .map((s) => {
      const oneHot = new Array(vocabSize).fill(0);
      oneHot[s] = 1;
      return oneHot;
    })
    .flat();

  // Initialize weights with smaller values
  const embeddings = g
    .tensor([embedSize, vocabSize], "embeddings")
    .set(heInit([embedSize, vocabSize], 0.1));

  const X = g.tensor([batchSize * seqLength, vocabSize], "X").set(xOneHot);

  // Convert labels to one-hot
  const labelTensor = g.tensor([batchSize, vocabSize], "Y").set(
    labels
      .map((l) => {
        const oneHot = new Array(vocabSize).fill(0);
        oneHot[l] = 1;
        return oneHot;
      })
      .flat(),
  );

  // Attention weights - smaller initialization
  const Wq = g.tensor([embedSize, embedSize], "Wq").set(heInit([embedSize, embedSize], 0.1));
  const Wk = g.tensor([embedSize, embedSize], "Wk").set(heInit([embedSize, embedSize], 0.1));
  const Wv = g.tensor([embedSize, embedSize], "Wv").set(heInit([embedSize, embedSize], 0.1));

  // Output projection
  const Wout = g.tensor([embedSize, vocabSize], "Wout").set(heInit([embedSize, vocabSize], 0.1));
  const bout = g.tensor([vocabSize], "bout").fill(0);

  // 1. Embed input
  const embedded = matmul(X, transpose(embeddings));

  // 2. Self-attention
  const Q = matmul(embedded, Wq);
  const K = matmul(embedded, Wk);
  const V = matmul(embedded, Wv);

  const Q_reshaped = reshape(Q, [batchSize * numHeads, seqLength * headDim]);
  const K_reshaped = reshape(K, [batchSize * numHeads, seqLength * headDim]);
  const V_reshaped = reshape(V, [batchSize * numHeads, seqLength * headDim]);

  // Add small epsilon to prevent division by zero
  const scores = matmul(Q_reshaped, transpose(K_reshaped));
  const scaled = div(scores, add(sqrt(headDim), 0.001));
  const weights = softmax(scaled);

  const attended = matmul(weights, V_reshaped);
  const output = reshape(attended, [batchSize * seqLength, embedSize]);
  const lastHidden = reshape(output, [batchSize, embedSize]);

  // Output projection
  const logits = add(matmul(lastHidden, Wout), bout);
  const predictions = softmax(logits);

  const loss = g.output(crossEntropy(predictions, labelTensor));
  g.compile(loss, [batchSize]);

  return executeEpoch({
    tensors: [embeddings, Wq, Wk, Wv, Wout, bout],
    graph: g,
    predictions,
  });
};
