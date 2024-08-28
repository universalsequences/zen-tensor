import { Context } from "./context";
import { memo } from "./memo";
import { OpType, ASTNode, Arg, DataType } from "./zen";
import { toScalar } from "./zen";

const binaryOp = (name: string, op: string) => (x: Arg, y: Arg) =>
  memo(
    (context: Context): ASTNode => {
      context = context.useContext(OpType.Regular);
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

      const [variableName] = context.useVariables(`${name}_result`);

      let code = `let ${variableName} = ${toScalar(_x)} ${op} ${toScalar(_y)};`;
      return context.emit(variableName, code, OpType.Regular, outputShape, _x, _y);
    },
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

export const add = binaryOp("add", "+");
export const mult = binaryOp("mult", "*");
export const sub = binaryOp("sub", "-");
export const div = binaryOp("div", "/");

export const reduce =
  (op: string) =>
  (x: Arg) =>
  (context: Context): ASTNode => {
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
  };

export const sum = reduce("+");

export const mean = (x: Arg) =>
  memo((context: Context): ASTNode => {
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
  }, x);

export const func = (name: string) => {
  return (freq: Arg) => {
    return memo((context: Context): ASTNode => {
      context = context.useContext(OpType.Regular);
      const [variableName] = context.useVariables(`${name}_result`);
      const _freq = context.gen(freq);
      const code = `
let ${variableName} = ${name}(${toScalar(_freq)});
  `;
      return context.emit(variableName, code, OpType.Regular, _freq.shape, _freq);
    }, freq);
  };
};

export const sine = func("sin");
export const log2 = func("log2");

export const matmul =
  (a: Arg, b: Arg) =>
  memo((context: Context): ASTNode => {
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
  }, a, b);
