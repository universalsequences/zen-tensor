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
      const shapeX = _x.shape;
      const shapeY = _y.shape;
      let outputShape: number[];

      if (arraysEqual(shapeX, shapeY)) {
        outputShape = shapeX;
      } else if (isScalar(shapeX) || isScalar(shapeY)) {
        outputShape = isScalar(shapeX) ? shapeY : shapeX;
      } else if (shapeX.length === shapeY.length + 1 && arraysEqual(shapeX.slice(1), shapeY)) {
        // This handles the case of adding a matrix and a vector
        outputShape = shapeX;
      } else {
        throw new Error(`Incompatible shapes for ${name} operation: ${shapeX} and ${shapeY}`);
      }

      let code: string;
      if (arraysEqual(shapeX, shapeY) || isScalar(shapeX) || isScalar(shapeY)) {
        code = `let ${variableName} = ${toScalar(_x)} ${op} ${toScalar(_y)};`;
      } else {
        // Broadcasting for matrix + vector
        const batchSize = shapeX[0];
        const vectorSize = shapeY[0];
        code = `
// broadcast
          let batchIndex = index / ${vectorSize}u;
          let vectorIndex = index % ${vectorSize}u;
          let ${variableName} = ${toScalar(_x)} ${op} ${toScalar(_y, DataType.Scalar, "vectorIndex")};
        `;
      }

      return context.emit(op, variableName, code, OpType.Regular, outputShape, _x, _y);
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

const grad = (
  node: ASTNode,
  gradOut: string,
  customLogic?: (
    dep: ASTNode,
    i: number,
    isBroadcasting: boolean,
    shapes: number[][],
  ) => {
    code: string;
    intermediateVariables?: string[];
  },
) => {
  let code = "";
  let intermediateVariables: string[] = [];
  const visited = new Set<string>();
  const shapes = node.dependencies.map((dep) => dep.shape);
  const isBroadcasting =
    shapes[0].length !== shapes[1].length || !arraysEqual(shapes[0], shapes[1]);

  for (let i = node.dependencies.length - 1; i >= 0; i--) {
    const dep = node.dependencies[i];
    if (visited.has(dep.gradientVariable)) continue;
    visited.add(dep.gradientVariable);

    const grad = customLogic
      ? customLogic(dep, i, isBroadcasting, shapes)
      : { code: `${dep.gradientVariable} += ${gradOut};\n` };

    code += grad.code;
    if (grad.intermediateVariables) {
      intermediateVariables.push(...grad.intermediateVariables);
    }
  }

  return {
    code,
    intermediateVariables: intermediateVariables.map((x) => trimIndex(x)),
  };
};

// removes the [index] from a string
export const trimIndex = (x: string) => {
  if (x.includes("[index]")) {
    return x.slice(0, x.length - "[index]".length);
  }
  return x;
};

export const add = binaryOp("add", "+", (node: ASTNode, gradOut: string) =>
  grad(node, gradOut, (dep, i, isBroadcasting, shapes) => {
    if (!isBroadcasting) {
      return {
        code: `${dep.gradientVariable} += ${gradOut};\n`,
        intermediateVariables: [gradOut],
      };
    } else {
      // Handle broadcasting case
      const [shape1, shape2] = shapes;
      if (shape1.length === shape2.length + 1 && arraysEqual(shape1.slice(1), shape2)) {
         // Matrix + Vector broadcasting
        const batchSize = shape1[0];
        const vectorSize = shape2[0];
        if (i === 0) {
          // Matrix
          return {
            code: `${dep.gradientVariable} += ${gradOut};\n`,
            intermediateVariables: [gradOut],
          };
        } else {
          // Vector (bias)
          let intermediate = `grad_${node.parent?.variable}_output`;
          return {
            code: `
              // broadcast
              let vectorIndex = index % ${vectorSize}u;
              var grad_sum = 0.0;
              for (var i = 0u; i < ${batchSize}u; i = i + 1u) {
                grad_sum += ${intermediate}[i * ${vectorSize}u + vectorIndex];
              }
              ${dep.gradientVariable} = grad_sum;
            `,
            intermediateVariables: [gradOut],
          };
        }
      } else {
        throw new Error(`Unsupported broadcasting case for shapes: ${shape1} and ${shape2}`);
      }
    }
  }),
);

export const sub = binaryOp("sub", "-", (node: ASTNode, gradOut: string) =>
  grad(node, gradOut, (dep, i) => ({
    code: `${dep.gradientVariable} += ${i === 0 ? gradOut : `-${gradOut}`};\n`,
  })),
);

export const v = (a: ASTNode) =>
  a.type === DataType.Tensor ? toScalar(a) : `${intermediate(a)}[index]`;

export const mult = binaryOp("mult", "*", (node: ASTNode, gradOut: string) => {
  return grad(node, gradOut, (dep, i) => {
    const otherDep = node.dependencies[1 - i];
    if (node.dependencies[0].variable === node.dependencies[1].variable) {
      // Handling the case where both dependencies are the same (e.g., b * b)
      const code = `
${dep.gradientVariable} += 2.0 * ${gradOut}*${v(otherDep)}
`;
      return {
        code,
        intermediateVariables: [gradOut, v(otherDep)],
      };
    } else {
      const code = `${dep.gradientVariable} += ${gradOut} * ${v(otherDep)};\n`;
      return {
        code,
        intermediateVariables: [gradOut, v(otherDep)],
      };
    }
  });
});

export const div = binaryOp("div", "/", (node: ASTNode, gradOut: string) =>
  grad(node, gradOut, (dep, i) => {
    if (i === 0) {
      // Gradient for the first operand (dividend)
      const code = `${dep.gradientVariable} += ${gradOut} / ${v(node.dependencies[1])};\n`;
      return {
        code,
        intermediateVariables: [gradOut, v(node.dependencies[i])],
      };
    } else {
      // Gradient for the second operand (divisor)
      const code = `${dep.gradientVariable} += -${gradOut} * ${v(node.dependencies[0])} / (${v(dep)} * ${v(dep)});\n`;
      return {
        code,
        intermediateVariables: [gradOut, v(node.dependencies[0]), v(dep)],
      };
    }
  }),
);

export const reduce = (op: string) => (x: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      const reductionContext = context.useContext(OpType.Reduction);
      const [variableName] = reductionContext.useVariables(`reduce_result`);
      const _x = reductionContext.gen(x);
      const code = `
    var ${variableName} = ${_x.variable}[0];
    for (var i = 1u; i < arrayLength(&${_x.variable}); i = i + 1u) {
      ${variableName} = ${variableName} ${op} ${toScalar(_x, DataType.Scalar, "i")};
    }
  `;
      return reductionContext.emit(
        `reduce.${op}`,
        variableName,
        code,
        OpType.Reduction,
        _x.shape,
        _x,
      );
    },
    (node: ASTNode) => {
      const inputVar = node.dependencies[0].gradientVariable;
      const gradientCode = `
    ${inputVar} += 1;
  `;
      return {
        code: gradientCode,
        intermediateVariables: [],
      };
    },
    x,
  );

export const sum = reduce("+");

export const mean = (x: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      const reductionContext = context.useContext(OpType.Reduction);
      const _x = reductionContext.gen(x);
      //const [sumVariable] = reductionContext.useVariables(`mean_sum`);
      //const [countVariable] = reductionContext.useVariables(`mean_count`);
      const [resultVariable] = reductionContext.useVariables(`mean_result`);
      const sumVariable = `${resultVariable}_sum`;
      const countVariable = `${resultVariable}_count`;

      const code = `
var ${sumVariable} = 0.0;
var ${countVariable} = 0u;
for (var i = 0u; i < arrayLength(&${_x.variable}); i = i + 1u) {
  ${sumVariable} = ${sumVariable} + ${_x.variable}[i];
  ${countVariable} = ${countVariable} + 1u;
}
let ${resultVariable} = ${sumVariable} / f32(${countVariable});
`;

      return reductionContext.emit(
        "reduce.mean",
        resultVariable,
        code,
        OpType.Reduction,
        _x.shape,
        _x,
      ); // Mean always outputs a single value
    },
    (node: ASTNode) => {
      const inputVar = node.dependencies[0].gradientVariable; // This is the gradient variable corresponding to the input of the sum operation
      const gradientCode = `
let ${node.variable}_length = arrayLength(&${trimIndex(v(node.dependencies[0]))});  // Total number of elements in x
    ${inputVar} += 1 / f32(${node.variable}_length);
  `;
      return {
        code: gradientCode,
        intermediateVariables: [trimIndex(v(node.dependencies[0]))],
      };
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
        return context.emit(name, variableName, code, OpType.Regular, _freq.shape, _freq);
      },
      (node: ASTNode, gradOut: string) => {
        const inputVar = node.dependencies[0].variable;
        const gradientCode = `
          let grad_${inputVar} = ${gradOut} * ${derivative}(${inputVar});
        `;
        return {
          code: gradientCode,
          intermediateVariables: [],
        };
      },
      freq,
    );
  };
};

export const sine = func("sin", "cos"); // sin'(x) = cos(x)
export const log2 = func("log2", "1.0 / (x * log(2.0))"); // d/dx(log2(x)) = 1 / (x * log(2))
