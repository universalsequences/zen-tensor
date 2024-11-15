import type { Arg, ASTNode } from "./zen";
import { memo } from "./memo";
import { OpType, type Context, getShape } from "./index";

const epsilon = 0.0001;
export const batchNorm = (x: Arg, gamma: Arg, beta: Arg) =>
  memo(
    (context: Context<ASTNode>) => {
      // Forward Pass
      const _x = context.gen(x);
      const _gamma = context.gen(gamma);
      const _beta = context.gen(beta);

      const xShape = getShape(_x);
      const gammaShape = getShape(_gamma);
      const betaShape = getShape(_beta);

      if (gammaShape[1] !== xShape[1] || betaShape[1] !== xShape[1]) {
        throw new Error(`Gamma and beta must have shape [1, ${xShape[1]}]`);
      }

      const [result, normalized] = context.useVariables("batch_norm_result", "normalized");
      const epsilonVar = epsilon;

      const normalizedCode = `
// Compute mean
var mean = 0.0;
for (var i = 0u; i < ${xShape[0]}u; i++) {
  mean += ${_x.variable}[i * ${xShape[1]} + index];
}
mean /= ${xShape[0]}f;

// Compute variance
var variance = 0.0;
for (var i = 0u; i < ${xShape[0]}u; i++) {
  let diff = ${_x.variable}[i * ${xShape[1]} + index] - mean;
  variance += diff * diff;
}
variance /= ${xShape[0]}f;
// Normalize
let ${normalized} = (${_x.variable}[index] - mean) / sqrt(variance + ${epsilonVar});
`;

      const dep = context.emit("normalized", result, normalizedCode, OpType.Regular, xShape, _x);

      const batchNormCode = `
// Apply gamma and beta
let ${result} = ${normalized} * ${_gamma.variable}[index % ${xShape[1]}] + ${_beta.variable}[index % ${xShape[1]}];
`;

      return context.emit(
        "batchNorm",
        result,
        batchNormCode,
        OpType.Regular,
        xShape,
        dep,
        _gamma,
        _beta,
      );
    },
    (node: ASTNode, gradOut: string) => {
      // Backward Pass
      const shape = getShape(node);
      const len = shape[0] * shape[1];
      const normalized = node.dependencies[0].variable;
      const gradCode = `
// Compute gradients
var grad_x = ${gradOut};
var grad_gamma = ${gradOut} * ${normalized};
var grad_beta = ${gradOut};

// Output gradients
${node.gradientVariable} = grad_x;
`;

      return {
        code: gradCode,
        intermediateVariables: [], // Add intermediates as needed
        gradientOutputs: [node.gradientVariable],
      };
    },
    x,
    gamma,
    beta,
  );
