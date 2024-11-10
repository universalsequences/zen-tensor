import { Tensor, NodeGen, TensorGraph } from "@/lib";

export interface EpochResult {
  loss: number;
  tensors: Map<string, Float32Array>;
  gradients: Map<string, Float32Array>;
  forward: Float32Array;
  predicition: number[] | undefined;
  computation: ASTNode | undefined;
}

export const executeEpoch =
  ({
    tensors,
    graph,
    predictions,
  }: {
    tensors: Tensor[];
    predictions: NodeGen;
    graph: TensorGraph;
  }) =>
  async (learningRate: number): Promise<EpochResult> => {
    const { forward, gradients } = await graph.run();
    for (const tensor of tensors) {
      tensor.learn(learningRate);
    }
    const tensorResults = new Map<string, Float32Array>();
    for (const [key, tensor] of graph.tensors.entries()) {
      tensorResults.set(key, tensor.val());
    }

    let sum = forward.reduce((a, b) => a + b, 0);
    const loss = sum / forward.length;
    let predictionResult;
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
    };
  };

export const heInit = (shape: [number, number], scale = 1) => {
  const fanIn = shape[0];
  return Array(shape[0] * shape[1])
    .fill(0)
    .map(() => (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn) * scale);
};
