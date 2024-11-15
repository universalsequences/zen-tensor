import { pow2, sqrt, trimIndex, v } from "./math";
import { memo } from "./memo";
import type { Arg, ASTNode } from "./zen";
import { OpType, matmul, div, type Context, sub, add, mult, transpose, getShape } from "./index";
import { constant } from "./constant";

const epsilon = 0.0001;
/*
export const layerNorm = (x: Arg, gamma: Arg, beta: Arg) => {
  return (context: Context<ASTNode>) => {
    const _x = context.gen(x);
    const _gamma = context.gen(gamma);
    const _beta = context.gen(beta);
    const xShape = getShape(_x);
    const gammaShape = getShape(_gamma);
    const betaShape = getShape(_beta);
    const batchSize = xShape[0];
    const featureSize = xShape[1];

    // Validate gamma and beta shapes
    if (gammaShape[1] !== featureSize || betaShape[1] !== featureSize) {
      throw new Error(`Gamma and beta must have shape [1, ${featureSize}]`);
    }

    const onesF = constant([1, featureSize], 1);
    const epsilonConstant = constant([batchSize, featureSize], epsilon);

    // Mean across feature dimension (note transpose difference from batchNorm)
    const mean = div(matmul(x, transpose(onesF)), featureSize);
    const broadcastMean = matmul(mean, onesF);

    // Variance across feature dimension
    const diffSquared = pow2(sub(x, broadcastMean));
    const variance = div(matmul(diffSquared, transpose(onesF)), featureSize);
    const broadcastVar = matmul(variance, onesF);

    // Normalize
    const normalized = div(sub(x, broadcastMean), sqrt(add(broadcastVar, epsilonConstant)));

    // Scale and shift
    const broadcastGamma = matmul(constant([batchSize, 1], 1), gamma);
    const broadcastBeta = matmul(constant([batchSize, 1], 1), beta);

    return add(mult(normalized, broadcastGamma), broadcastBeta)(context);
  };
};

*/

export const layerNormOld = (x: Arg, gamma: Arg, beta: Arg) =>
  memo(
    (_context: Context<ASTNode>) => {
      const context = _context.useContext(OpType.Regular);

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

      const [meanBuffer, varianceBuffer, normalized, result] = context.useVariables(
        "mean",
        "variance",
        "normalized",
        "layer_norm_result",
      );

      const featureSize = xShape[1];
      const epsilonVar = epsilon;

      // Compute mean
      const meanCode = `
// Compute mean across feature dimension
var ${meanBuffer} = 0.0;
for (var i = 0u; i < ${featureSize}u; i++) {
  ${meanBuffer} += ${_x.variable}[index / ${featureSize} * ${featureSize} + i];
}
${meanBuffer} /= ${featureSize}f;
`;

      const depMean = context.emit(
        "mean",
        meanBuffer,
        meanCode,
        OpType.Regular,
        [xShape[0], 1],
        _x,
      );

      // Compute variance
      const varianceCode = `
// Compute variance across feature dimension
var ${varianceBuffer} = 0.0;
for (var i = 0u; i < ${featureSize}u; i++) {
  let diff = ${_x.variable}[index / ${featureSize} * ${featureSize} + i] - ${meanBuffer};
  ${varianceBuffer} += diff * diff;
}
${varianceBuffer} /= ${featureSize}f;
`;

      const depVariance = context.emit(
        "variance",
        varianceBuffer,
        varianceCode,
        OpType.Regular,
        [xShape[0], 1],
        _x,
        depMean,
      );

      // Normalize
      const normalizedCode = `
// Normalize
let ${normalized} = (${_x.variable}[index] - ${meanBuffer}) / sqrt(${varianceBuffer} + ${epsilonVar});
`;

      const depNormalized = context.emit(
        "normalized",
        normalized,
        normalizedCode,
        OpType.Regular,
        xShape,
        depVariance,
        depMean,
        _x,
      );

      // Apply gamma and beta
      const layerNormCode = `
// Apply gamma and beta
let ${result} = ${normalized} * ${_gamma.variable}[index % ${featureSize}] + ${_beta.variable}[index % ${featureSize}];
`;

      return context.emit(
        "layerNorm",
        result,
        layerNormCode,
        OpType.Regular,
        xShape,
        depNormalized,
        _gamma,
        _beta,
      );
    },
    (node: ASTNode, gradOut: string) => {
      // Backward Pass
      const _x = node.dependencies[0];
      const _gamma = node.dependencies[1];
      const _beta = node.dependencies[2];
      const shape = getShape(node);
      const len = shape[0] * shape[1];
      const featureSize = shape[1];

      const normalized = node.dependencies[0];
      const variance = node.dependencies[0].dependencies[0];
      const mean = node.dependencies[0].dependencies[1];

      const gradCode = `
// Compute gradients
var grad_normalized = ${gradOut} * ${_gamma.variable}[index % ${featureSize}];
var variance = ${variance.variable}[index / ${featureSize}];
var mean = ${mean.variable}[index / ${featureSize}];

var grad_variance = 0.0;
var grad_mean = 0.0;
for (var i = 0u; i < ${featureSize}u; i++) {
  let diff = ${trimIndex(v(normalized))}[index / ${featureSize} * ${featureSize} + i] - mean;
  grad_variance += grad_normalized * diff * -0.5 / pow(variance + ${epsilon}, 1.5);
  grad_mean += grad_normalized * -1.0 / sqrt(variance + ${epsilon}) +
               grad_variance * -2.0 * diff / ${featureSize}f;
}

var grad_x = grad_normalized / sqrt(variance + ${epsilon}) +
             grad_variance * 2.0 * (${trimIndex(v(normalized))}[index] - mean) / ${featureSize}f +
             grad_mean / ${featureSize}f;

var grad_gamma = ${gradOut} * ${v(normalized)};
var grad_beta = ${gradOut};

// Output gradients
${normalized.gradientVariable} = grad_x;
${mean.gradientVariable} = grad_mean;
${_gamma.gradientVariable} += grad_gamma;
${_beta.gradientVariable} += grad_beta;
`;

      return {
        code: gradCode,
        intermediateVariables: [
          trimIndex(v(normalized)),
          mean.variable,
          variance.variable,
          trimIndex(_gamma.variable),
        ],
        gradientOutputs: [
          variance.gradientVariable,
          //mean.gradientVariable,
          normalized.gradientVariable,
          _gamma.gradientVariable,
          _beta.gradientVariable,
        ],
      };
    },
    x,
    gamma,
    beta,
  );

export const layerNorm = (x: Arg, gamma: Arg, beta: Arg) =>
  memo(
    (_context: Context<ASTNode>) => {
      const context = _context.useContext(OpType.Regular);

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

      const [meanBuffer, varianceBuffer, normalized, result] = context.useVariables(
        "mean",
        "variance",
        "normalized",
        "layer_norm_result",
      );

      const featureSize = xShape[1];
      const epsilonVar = epsilon;

      // Compute mean
      const meanCode = `
// Compute mean across feature dimension
var ${meanBuffer} = 0.0;
for (var i = 0u; i < ${featureSize}u; i++) {
  ${meanBuffer} += ${_x.variable}[index / ${featureSize} * ${featureSize} + i];
}
${meanBuffer} /= ${featureSize}f;
`;

      const depMean = context.emit(
        "mean",
        meanBuffer,
        meanCode,
        OpType.Regular,
        [xShape[0], 1],
        _x,
      );

      // Compute variance
      const varianceCode = `
// Compute variance across feature dimension
var ${varianceBuffer} = 0.0;
for (var i = 0u; i < ${featureSize}u; i++) {
  let diff = ${_x.variable}[index / ${featureSize} * ${featureSize} + i] - ${meanBuffer};
  ${varianceBuffer} += diff * diff;
}
${varianceBuffer} /= ${featureSize}f;
`;

      const depVariance = context.emit(
        "variance",
        varianceBuffer,
        varianceCode,
        OpType.Regular,
        [xShape[0], 1],
        _x,
        depMean,
      );

      // Normalize
      const normalizedCode = `
// Normalize
let ${normalized} = (${_x.variable}[index] - ${meanBuffer}) / sqrt(${varianceBuffer} + ${epsilonVar});
`;

      const depNormalized = context.emit(
        "normalized",
        normalized,
        normalizedCode,
        OpType.Regular,
        xShape,
        depVariance,
        depMean,
        _x,
      );

      // Apply gamma and beta
      const layerNormCode = `
// Apply gamma and beta
let ${result} = ${normalized} * ${_gamma.variable}[index % ${featureSize}] + ${_beta.variable}[index % ${featureSize}];
`;

      return context.emit(
        "layerNorm",
        result,
        layerNormCode,
        OpType.Regular,
        xShape,
        depNormalized,
        _gamma,
        _beta,
      );
    },
    (node: ASTNode, gradOut: string) => {
      // Backward Pass
      const _x = node.dependencies[0];
      const _gamma = node.dependencies[1];
      const _beta = node.dependencies[2];
      const shape = getShape(node);
      const featureSize = shape[1];

      const normalized = node.dependencies[0];
      const meanBuffer = node.dependencies[0].dependencies[1];

      const gradCode = `
  // Compute gradients
  var grad_normalized = ${gradOut} * ${_gamma.variable}[index % ${featureSize}];
  var grad_gamma = ${gradOut} * ${v(normalized)};
  var grad_beta = ${gradOut};
  
  // Recompute variance using mean
  var variance = 0.0;
  for (var i = 0u; i < ${featureSize}u; i++) {
    let diff = ${trimIndex(v(normalized))}[index / ${featureSize} * ${featureSize} + i] - ${v(meanBuffer)};
    variance += diff * diff;
  }
  variance /= ${featureSize}f;
  
  // Compute gradients for variance and mean
  var grad_variance = 0.0;
  var grad_mean = 0.0;
  for (var i = 0u; i < ${featureSize}u; i++) {
    let diff = ${trimIndex(v(normalized))}[index / ${featureSize} * ${featureSize} + i] - ${v(meanBuffer)};
    grad_variance += grad_normalized * diff * -0.5 / pow(variance + ${epsilon}, 1.5);
    grad_mean += grad_normalized * -1.0 / sqrt(variance + ${epsilon}) +
                 grad_variance * -2.0 * diff / ${featureSize}f;
  }
  
  // Compute gradient for input
  var grad_x = grad_normalized / sqrt(variance + ${epsilon}) +
               grad_variance * 2.0 * (${trimIndex(v(normalized))}[index] - ${v(meanBuffer)}) / ${featureSize}f +
               grad_mean / ${featureSize}f;
  
  // Output gradients
  `;

      return {
        code: gradCode,
        intermediateVariables: [
          trimIndex(v(meanBuffer)),
          trimIndex(v(normalized)),
          trimIndex(_gamma.variable),
        ],

        gradientOutputs: [_x.gradientVariable, _gamma.gradientVariable, _beta.gradientVariable],
      };
    },
    x,
    gamma,
    beta,
  );
