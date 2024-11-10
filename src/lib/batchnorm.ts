import { Context } from "./context";
import { trimIndex, v } from "./math";
import { memo } from "./memo";
import { emitIntermediate } from "./utils";
import { Arg, ASTNode, DataType, intermediate, OpType, toScalar } from "./zen";
import { matmul, div, sub, add, mult } from "./index";
import { constant } from "./constant";

const epsilon = 0.0001;
export const batchNorm = (x: Arg, gamma: Arg, beta: Arg) => {
  // Let's say our core operators are:
  // matmul, add, mult (element-wise multiplication),
  // div (element-wise division), sub (element-wise subtraction)
  // sum (across specified axis)

  // For mean: we need to sum across batch dimension and divide by batch size
  // If we have a ones vector of shape [batchSize, 1], we can use matmul to sum
  const batchSize = x.shape[0];
  const ones = constant([batchSize, 1], 1);

  // Mean = (1/N) * sum(x) for each feature
  // matmul with ones sums across batch dimension
  const mean = div(matmul(transpose(x), ones), batchSize);

  // For variance: (x - mean)^2
  // First broadcast mean to same shape as x
  const broadcastMean = matmul(ones, transpose(mean));
  const variance = div(matmul(transpose(pow(sub(x, broadcastMean), 2)), ones), batchSize);

  // Normalize: (x - mean) / sqrt(var + eps)
  const normalized = div(sub(x, broadcastMean), sqrt(add(variance, epsilon)));

  // Scale and shift
  return add(mult(normalized, gamma), beta);
};

/*
However, this highlights that we might be missing some core operators that would make this more efficient:

1. We need broadcasting capability
2. We need elementwise power/square
3. We need sum across specific dimensions
4. We need sqrt

Would you like me to:
1. List out what core operators we'd need to add to make this work efficiently?
2. Show alternative implementations using different sets of core operators?
3. Break down which parts of BatchNorm require which types of operations?

The exact implementation would depend on what core operators are available in your framework. What operators do you currently have available?����������������
*/
