import { Context } from "./context";
import { trimIndex, v } from "./math";
import { memo } from "./memo";
import { getShape } from "./reshape";
import { emitIntermediate } from "./utils";
import { Arg, ASTNode, DataType, intermediate, OpType, toScalar } from "./zen";

export const relu = (x: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables(`relu_result`);

      const _x = context.gen(x);

      let code = `let ${res} = max(0.0, ${v(_x)});`;
      return context.emit("relu", res, code, OpType.Regular, getShape(_x), _x);
    },
    (node: ASTNode, gradOut: string) => {
      const gradCode = `
        ${node.gradientVariable} = select(0.0, ${gradOut}, ${v(node.dependencies[0])} > 0.0);
      `;
      return {
        code: gradCode,
        intermediateVariables: emitIntermediate(node),
        gradientOutputs: [node.gradientVariable],
      };
    },
    x,
  );

export const sigmoid = (x: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables("sigmoid_result");
      const _x = context.gen(x);
      let code = `let ${res} = 1.0 / (1.0 + exp(-${v(_x)}));`;
      return context.emit("sigmoid", res, code, OpType.Regular, getShape(_x), _x);
    },
    (node: ASTNode, gradOut: string) => {
      const inputVar = node.dependencies[0].variable;
      const gradCode = `
        let sigmoid_${inputVar} = 1.0 / (1.0 + exp(-${v(node.dependencies[0])}));
        ${node.gradientVariable} = ${gradOut} * sigmoid_${inputVar} * (1.0 - sigmoid_${inputVar});
      `;
      return {
        code: gradCode,
        intermediateVariables: emitIntermediate(node),
        gradientOutputs: [node.gradientVariable],
      };
    },
    x,
  );

export const leakyRelu = (x: Arg, alpha: number = 0.01) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables(`leaky_relu_result`);
      const _x = context.gen(x);
      let code = `let ${res} = select(
${alpha} * ${context.getReference(_x)},
${context.getReference(_x)},
${context.getReference(_x)} > 0.0);`;
      return context.emit("leakyRelu", res, code, OpType.Regular, getShape(_x), _x);
    },
    (node: ASTNode, gradOut: string) => {
      const code = `
        ${node.gradientVariable} = select(
${alpha} * ${gradOut},
${gradOut}, ${v(node.dependencies[0])} > 0.0);
      `;
      return {
        code,
        intermediateVariables: emitIntermediate(node),
        gradientOutputs: [node.gradientVariable],
      };
    },
    x,
    alpha,
  );

export const softmax = (input: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      // Forward Pass
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables("softmax");
      const _input = context.gen(input);
      const shape = getShape(_input);

      const [batchSize, numClasses] = shape.length === 2 ? shape : [1, shape[0]]; // Handle both 1D and 2D cases
      const len = batchSize * numClasses;

      const code = `
      let row = index / ${numClasses}u; // Determine the row
      let col = index % ${numClasses}u; // Determine the column
  
      // Compute the maximum value for the current row
      var max_val = -1e20; // Initialize to a very small number
      for (var i = 0u; i < ${numClasses}u; i++) {
        let idx = row * ${numClasses}u + i;
        max_val = max(max_val, ${_input.variable}[idx]);
      }
  
      // Compute the sum of exponentials for the current row
      var exp_sum = 0.0;
      for (var i = 0u; i < ${numClasses}u; i++) {
        let idx = row * ${numClasses}u + i;
        exp_sum += exp(${_input.variable}[idx] - max_val);
      }
  
      // Compute the softmax value for the current element
      var ${res} = 0.0;
      if (index < ${len}u) {
        let numerator = exp(${toScalar(_input)} - max_val);
        ${res} = numerator / exp_sum;
      }
    `;
      return context.emit("softmax", res, code, OpType.Regular, getShape(_input), _input);
    },
    (node: ASTNode, gradOut: string) => {
      // Backward Pass
      const softmaxVar = intermediate(node);
      const shape = getShape(node);

      const [batchSize, numClasses] = shape.length === 2 ? shape : [1, shape[0]]; // Handle both 1D and 2D cases
      const len = batchSize * numClasses;

      const gradCode = `
      let row = index / ${numClasses}u; // Determine the row
      let col = index % ${numClasses}u; // Determine the column
  
      if (index < ${len}u) {
        var grad_accumulator = 0.0;
        var grad_softmax = ${gradOut}; // Gradient of the output wrt loss
  
        // Compute the gradient for this thread
        for (var i = 0u; i < ${numClasses}u; i++) {
          let idx = row * ${numClasses}u + i;
          let kronecker_delta = select(0.0, 1.0, i == col);
          grad_accumulator += grad_softmax * ${softmaxVar}[idx] * (kronecker_delta - ${softmaxVar}[index]);
        }
  
        // Add accumulated gradient
        grad_softmax += grad_accumulator;
      }
    `;
      return {
        code: gradCode,
        intermediateVariables: [trimIndex(softmaxVar)],
        gradientOutputs: [node.gradientVariable],
      };
    },
    input,
  );

export const tanh = (x: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables("tanh_result");
      const _x = context.gen(x);
      const code = `
        let exp2x = exp(2.0 * ${context.getReference(_x)});
        let ${res} = (exp2x - 1.0) / (exp2x + 1.0);
      `;
      return context.emit("tanh", res, code, OpType.Regular, getShape(_x), _x);
    },
    (node: ASTNode, gradOut: string) => {
      // d/dx tanh(x) = 1 - tanh²(x)
      const inputVar = node.dependencies[0].variable;
      const gradCode = `
        let exp2x_${inputVar} = exp(2.0 * ${v(node.dependencies[0])});
        let tanh_${inputVar} = (exp2x_${inputVar} - 1.0) / (exp2x_${inputVar} + 1.0);
        ${node.gradientVariable} = ${gradOut} * (1.0 - tanh_${inputVar} * tanh_${inputVar});
      `;
      return {
        code: gradCode,
        intermediateVariables: emitIntermediate(node),
        gradientOutputs: [node.gradientVariable],
      };
    },
    x,
  );
