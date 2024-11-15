import type { Tensor, NodeGen, TensorGraph, ASTNode } from "@/lib";

export interface EpochResult {
  loss: number;
  tensors: Map<string, Float32Array>;
  gradients: Map<string, Float32Array>;
  forward: Float32Array;
  predicition: number[] | undefined;
  computation: ASTNode | undefined;
  learningTime: number;
}

export const executeEpoch =
  ({
    tensors,
    graph,
    predictions,
    entropy,
    ignoreZeroLoss = false,
  }: {
    tensors: Tensor[];
    predictions: NodeGen;
    graph: TensorGraph;
    entropy?: NodeGen;
    ignoreZeroLoss?: boolean;
  }) =>
  async (learningRate: number): Promise<EpochResult> => {
    const { forward, gradients } = await graph.run();
    const a = new Date().getTime();
    for (const tensor of tensors) {
      tensor.learn(learningRate);
    }
    const b = new Date().getTime();
    const tensorResults = new Map<string, Float32Array>();
    for (const [key, tensor] of graph.tensors.entries()) {
      tensorResults.set(key, tensor.val());
    }

    let sum = forward.reduce((a, b) => a + b, 0);
    let loss = sum / forward.length;
    if (entropy?.node?.result) {
      const results = await entropy.node?.result();
      if (ignoreZeroLoss) {
        const nonZeroResults = results.filter((x) => x !== 0);
        loss = nonZeroResults.reduce((a, b) => a + b, 0) / nonZeroResults.length;
        console.log("loss results=", results, nonZeroResults, loss);
      } else {
        sum = results.reduce((a, b) => a + b, 0);
        loss = sum / results.length;
        console.log("lsos results=", results);
      }
    }
    let predictionResult: number[] | undefined;
    if (predictions.node?.result) {
      predictionResult = await predictions.node.result();
    }

    return {
      loss,
      tensors: tensorResults,
      forward,
      gradients,
      predicition: predictionResult,
      computation: predictions.node,
      learningTime: b - a,
    };
  };

export const heInit = (shape: [number, number], scale = 1) => {
  const fanIn = shape[0];
  return Array(shape[0] * shape[1])
    .fill(0)
    .map(() => (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn) * scale);
};
