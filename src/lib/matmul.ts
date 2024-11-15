import { memo } from "./memo";
import type { Context } from "./context";
import { OpType, type ASTNode, type Arg, toScalar } from "./zen";
import { emitIntermediate } from "./utils";
import { getIndex, getShape } from "./reshape";

export const matmul = (a: Arg, b: Arg) =>
  memo(
    // Forward Pass
    (_context: Context<ASTNode>): ASTNode => {
      const context = _context.useContext(OpType.Reduction);
      const _a = context.gen(a);
      const _b = context.gen(b);
      const shapeA = getShape(_a); //.shape;
      const shapeB = getShape(_b); //.shape;
      // Check if shapes are compatible for matrix multiplication
      if (shapeA.length !== 2 || shapeB.length !== 2 || shapeA[1] !== shapeB[0]) {
        throw new Error(`Incompatible shapes for matrix multiplication: ${shapeA} and ${shapeB}`);
      }
      const outputShape = [shapeA[0], shapeB[1]];

      const [resultVar] = context.useVariables("matmul");
      const [sum, M, N, K, row, col] = [
        `sum_${resultVar}`,
        `M_${resultVar}`,
        `N_${resultVar}`,
        `K_${resultVar}`,
        `row_${resultVar}`,
        `col_${resultVar}`,
      ];
      const code = `
let ${M} = ${shapeA[0]}u;
let ${N} = ${shapeB[1]}u;
let ${K} = ${shapeA[1]}u;
let ${row} = index / ${N};
let ${col} = index % ${N};
var ${sum} = 0.0;
for (var k = 0u; k < ${K}; k = k + 1u) {
  let a_idx = ${row} * ${K} + k;
  let b_idx = k * ${N} + ${col};
  ${sum} = ${sum} + ${toScalar(_a, "a_idx")} * ${toScalar(_b, "b_idx")};
}
let ${resultVar} = ${sum};
      `;
      return context.emit("matmul", resultVar, code, OpType.Reduction, outputShape, _a, _b);
    },
    // Back Propagation
    (node: ASTNode, gradOut: string) => {
      const [M, N, K] = [
        `${node.gradientVariable}_M`,
        `${node.gradientVariable}_N`,
        `${node.gradientVariable}_K`,
      ];
      // TODO - determine root cause of why this hack is needed, and then remove
      const parentGrad =
        node.parent?.operation === "output"
          ? gradOut
          : node.parent?.context !== node.context &&
              node.parent?.gradientVariable !== "grad_bce_result1"
            ? `${node.parent?.gradientVariable}_intermediate_output[grad_out_idx]`
            : `${node.parent?.gradientVariable}_output[grad_out_idx]`;
      const aVar = node.dependencies[0].variable;
      const bVar = node.dependencies[1].variable;
      const gradCode = `
// gradient starting from node = ${node.variable} --- deps: ${node.dependencies.map((x) => x.variable).join("+")}
let ${M} = ${getShape(node.dependencies[0])[0]}u;
let ${N} = ${getShape(node.dependencies[1])[1]}u;
let ${K} = ${getShape(node.dependencies[0])[1]}u;
let row = index / ${N};
let col = index % ${K};

// Gradient for A
var grad_a_sum = 0.0;
for (var n = 0u; n < ${N}; n = n + 1u) {
  let grad_out_idx = row * ${N} + n;
  let b_idx = ${getIndex(node.dependencies[1], "col", "n")};
  grad_a_sum += ${parentGrad} * ${bVar}[b_idx];
}
${node.dependencies[0].gradientVariable} = grad_a_sum;

// Gradient for B
var grad_b_sum = 0.0;
for (var m = 0u; m < ${M}; m = m + 1u) {
  let grad_out_idx = m * ${N};
  let a_idx = ${getIndex(node.dependencies[0], "m", "index")};
  grad_b_sum += ${parentGrad} * ${aVar}[a_idx];
}
${node.dependencies[1].gradientVariable} = grad_b_sum;
`;
      const intermediateVariables =
        node.parent?.context !== node.context
          ? Array.from(
              new Set([parentGrad.slice(0, parentGrad.indexOf("[")), ...emitIntermediate(node)]),
            )
          : emitIntermediate(node);

      return {
        code: gradCode,
        intermediateVariables: intermediateVariables,
        gradientOutputs: node.dependencies.map((x) => x.gradientVariable),
      };
    },
    a,
    b,
  );
