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
      const [res] = context.useVariables(`sigmoid_result`);
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
      const [res, max, sum] = context.useVariables("softmax", "max", "sum");
      const _input = context.gen(input);
      const shape = getShape(_input);
      const len = shape[shape.length - 1]; // Assuming last dimension is the one we're applying softmax to

      let code = `
// Find max for numerical stability
var ${max} = ${_input.variable}[0];
for (var i = 1u; i < ${len}u; i++) {
  ${max} = max(${max}, ${_input.variable}[i]);
}

// Compute exp and sum
var ${sum} = 0.0;
for (var i = 0u; i < ${len}u; i++) {
${sum} += exp(${toScalar(_input, "i")} - ${max});
}

// Compute softmax for this specific index
let ${res} = exp(${toScalar(_input)} - ${max}) / ${sum};
`;
      return context.emit("softmax", res, code, OpType.Regular, getShape(_input), _input);
    },
    (node: ASTNode, gradOut: string) => {
      // Backward Pass
      const softmaxVar = intermediate(node);
      const shape = getShape(node);
      const len = shape[shape.length - 1];

      const gradCode = `
for (var i = 0u; i < ${len}u; i++) {
let kronecker_delta = select(0.0, 1.0, i == index);
  ${node.gradientVariable} += ${gradOut} * ${softmaxVar}[i] * (kronecker_delta - ${softmaxVar}[index]);
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
      const [res] = context.useVariables(`tanh_result`);
      const _x = context.gen(x);
      let code = `
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
