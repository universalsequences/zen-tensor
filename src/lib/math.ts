import { BGen } from "./back";
import { intermediate } from "./zen";
import { Context } from "./context";
import { memo } from "./memo";
import { OpType, ASTNode, Arg, DataType } from "./zen";
import { toScalar } from "./zen";

const binaryOp = (name: string, op: string, backwards: BGen) => (x: Arg, y: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      context = context.useContext(OpType.Regular);
      const [variableName] = context.useVariables(`${name}_result`);

      const _x = context.gen(x);
      const _y = context.gen(y);

      // get shapes on args
      const shapeX = _x.shape;
      const shapeY = _y.shape;

      // Determine output shape
      let outputShape: number[];
      if (arraysEqual(shapeX, shapeY)) {
        outputShape = shapeX;
      } else if (isScalar(shapeX) || isScalar(shapeY)) {
        outputShape = isScalar(shapeX) ? shapeY : shapeX;
      } else {
        throw new Error(`Incompatible shapes for ${name} operation: ${shapeX} and ${shapeY}`);
      }

      let code = `let ${variableName} = ${toScalar(_x)} ${op} ${toScalar(_y)};`;
      return context.emit(variableName, code, OpType.Regular, outputShape, _x, _y);
    },
    backwards,
    x,
    y,
  );

// Helper functions
function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function isScalar(shape: number[]): boolean {
  return shape.length === 1 && shape[0] === 1;
}

// TODO - each op should return a list of "inputs" needed by the kernel
// for example, any intermediate values or input tensors
const grad = (
  op: string,
  node: ASTNode,
  gradOut: string,
  customLogic?: (dep: ASTNode, i: number) => string,
) => {
  let code = "";
  const visited = new Set<string>();
  for (let i = node.dependencies.length - 1; i >= 0; i--) {
    const dep = node.dependencies[i];
    if (visited.has(dep.gradientVariable)) continue;
    visited.add(dep.gradientVariable);

    // Use custom logic if provided, otherwise default to default gradient calculation
    const gradCode = customLogic
      ? customLogic(dep, i)
      : `let ${dep.gradientVariable} += ${gradOut};\n`;
    code += gradCode;
  }
  return code;
};

export const add = binaryOp("add", "+", (node: ASTNode, gradOut: string) =>
  grad("+", node, gradOut, (dep) => `${dep.gradientVariable} += ${gradOut};\n`),
);

export const sub = binaryOp("sub", "-", (node: ASTNode, gradOut: string) =>
  grad(
    "-",
    node,
    gradOut,
    (dep, i) => `${dep.gradientVariable} += ${i === 0 ? gradOut : `-${gradOut}`};\n`,
  ),
);

export const v = (a: ASTNode) => (a.type === DataType.Tensor ? toScalar(a) : `${intermediate(a)}[index]`);

export const mult = binaryOp("mult", "*", (node: ASTNode, gradOut: string) => {
  return grad("*", node, gradOut, (dep, i) => {
    const otherDep = node.dependencies[1 - i];
    if (node.dependencies[0].variable === node.dependencies[1].variable) {
      // Handling the case where both dependencies are the same (e.g., b * b)
      return `
${dep.gradientVariable} += 2.0 * ${gradOut}*${v(otherDep)}
`;
    } else {
      return `${dep.gradientVariable} += ${gradOut} * ${v(otherDep)};\n`;
    }
  });
});

export const div = binaryOp("div", "/", (node: ASTNode, gradOut: string) =>
  grad("/", node, gradOut, (dep, i) => {
    if (i === 0) {
      // Gradient for the first operand (dividend)
      return `${dep.gradientVariable} += ${gradOut} / ${v(node.dependencies[1])};\n`;
    } else {
      // Gradient for the second operand (divisor)
      return `${dep.gradientVariable} += -${gradOut} * ${v(node.dependencies[0])} / (${v(dep)} * ${v(dep)});\n`;
    }
  }),
);

export const reduce = (op: string) => (x: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      const reductionContext = context.useContext(OpType.Reduction);
      const _x = reductionContext.gen(x);
      const [variableName] = reductionContext.useVariables(`reduce_result`);
      const code = `
    var ${variableName} = ${_x.variable}[0];
    for (var i = 1u; i < arrayLength(&${_x.variable}); i = i + 1u) {
      ${variableName} = ${variableName} ${op} ${toScalar(_x, DataType.Scalar, "i")};
    }
  `;
      return reductionContext.emit(variableName, code, OpType.Reduction, _x.shape, _x);
    },
    (node: ASTNode, gradOut: string) => {
      const inputVar = node.dependencies[0].variable;
      const gradientCode = `
        for (var i = 0u; i < arrayLength(&${inputVar}); i = i + 1u) {
          grad_${inputVar}[i] += ${gradOut};
        }
      `;
      return gradientCode;
    },
    x,
  );

export const sum = reduce("+");

export const mean = (x: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      const reductionContext = context.useContext(OpType.Reduction);
      const _x = reductionContext.gen(x);
      const [sumVariable] = reductionContext.useVariables(`mean_sum`);
      const [countVariable] = reductionContext.useVariables(`mean_count`);
      const [resultVariable] = reductionContext.useVariables(`mean_result`);

      const code = `
var ${sumVariable} = 0.0;
var ${countVariable} = 0u;
for (var i = 0u; i < arrayLength(&${_x.variable}); i = i + 1u) {
  ${sumVariable} = ${sumVariable} + ${_x.variable}[i];
  ${countVariable} = ${countVariable} + 1u;
}
let ${resultVariable} = ${sumVariable} / f32(${countVariable});
`;

      return reductionContext.emit(resultVariable, code, OpType.Reduction, _x.shape, _x); // Mean always outputs a single value
    },
    (node: ASTNode, gradOut: string) => {
      const inputVar = node.dependencies[0].variable;
      const [countVariable] = node.context.useVariables(`mean_count`);
      const gradientCode = `
        for (var i = 0u; i < arrayLength(&${inputVar}); i = i + 1u) {
          grad_${inputVar}[i] += ${gradOut} / f32(${countVariable});
        }
      `;
      return gradientCode;
    },
    x,
  );

export const func = (name: string, derivative: string) => {
  return (freq: Arg) => {
    return memo(
      (context: Context<ASTNode>): ASTNode => {
        context = context.useContext(OpType.Regular);
        const [variableName] = context.useVariables(`${name}_result`);
        const _freq = context.gen(freq);
        const code = `
let ${variableName} = ${name}(${toScalar(_freq)});
  `;
        return context.emit(variableName, code, OpType.Regular, _freq.shape, _freq);
      },
      (node: ASTNode, gradOut: string) => {
        const inputVar = node.dependencies[0].variable;
        const gradientCode = `
          let grad_${inputVar} = ${gradOut} * ${derivative}(${inputVar});
        `;
        return gradientCode;
      },
      freq,
    );
  };
};

export const sine = func("sin", "cos"); // sin'(x) = cos(x)
export const log2 = func("log2", "1.0 / (x * log(2.0))"); // d/dx(log2(x)) = 1 / (x * log(2))

export const matmul = (a: Arg, b: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      context = context.useContext(OpType.Reduction);
      const _a = context.gen(a);
      const _b = context.gen(b);

      const shapeA = _a.shape;
      const shapeB = _b.shape;

      // Check if shapes are compatible for matrix multiplication
      if (shapeA.length !== 2 || shapeB.length !== 2 || shapeA[1] !== shapeB[0]) {
        throw new Error(`Incompatible shapes for matrix multiplication: ${shapeA} and ${shapeB}`);
      }

      const outputShape = [shapeA[0], shapeB[1]];
      const [resultVar, sum, M, N, K, row, col] = context.useVariables(
        "matmul",
        "sum",
        "M",
        "N",
        "K",
        "row",
        "col",
      );

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
  ${sum} = ${sum} + ${toScalar(_a, DataType.Scalar, "a_idx")} * ${toScalar(_b, DataType.Scalar, "b_idx")};
}

let ${resultVar} = ${sum};
  `;

      let m = context.emit(resultVar, code, OpType.Reduction, outputShape, _a, _b);
      return m;
    },
    (node: ASTNode, gradOut: string) => {
      const gradA = `
      for (var k = 0u; k < ${shapeA[1]}u; k = k + 1u) {
        let a_idx = ${M} * k;
        grad_${node.dependencies[0].variable}[a_idx] += ${gradOut} * ${node.dependencies[1].variable}[k * ${N} + col];
      }`;

      const gradB = `
      for (var k = 0u; k < ${shapeA[1]}u; k = k + 1u) {
        let b_idx = k * ${N} + ${col};
        grad_${node.dependencies[1].variable}[b_idx] += ${gradOut} * ${node.dependencies[0].variable}[row * ${K} + k];
      }`;

      return gradA + gradB;
    },
    a,
    b,
  );
