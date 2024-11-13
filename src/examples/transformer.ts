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
