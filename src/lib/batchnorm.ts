import { pow2, sqrt } from "./math";
import { Arg, ASTNode } from "./zen";
import { matmul, div, Context, sub, add, mult, transpose, getShape } from "./index";
import { constant } from "./constant";

const epsilon = 0.0001;
export const batchNorm = (x: Arg, gamma: Arg, beta: Arg) => {
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

    const onesB = constant([batchSize, 1], 1);
    const epsilonConstant = constant([batchSize, featureSize], epsilon);
    const mean = div(matmul(transpose(x), onesB), batchSize);
    const broadcastMean = matmul(onesB, transpose(mean));
    const diffSquared = pow2(sub(x, broadcastMean));
    const variance = div(matmul(transpose(diffSquared), onesB), batchSize);
    const broadcastVar = matmul(onesB, transpose(variance));
    const normalized = div(sub(x, broadcastMean), sqrt(add(broadcastVar, epsilonConstant)));
    const broadcastGamma = matmul(onesB, gamma);
    const broadcastBeta = matmul(onesB, beta);
    return add(mult(normalized, broadcastGamma), broadcastBeta)(context);
  };
};
