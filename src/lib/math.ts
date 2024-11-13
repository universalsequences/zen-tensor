import { BGen } from "./back";
import { BackPropagationOutput, intermediate } from "./zen";
import { Context } from "./context";
import { memo } from "./memo";
import { OpType, ASTNode, Arg, DataType } from "./zen";
import { toScalar } from "./zen";
import { emitIntermediate } from "./utils";
import { getShape } from "./reshape";

const binaryOp = (name: string, op: string, backwards: BGen) => (x: Arg, y: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      context = context.useContext(OpType.Regular);
      const [variableName] = context.useVariables(`${name}_result`);
      const _x = context.gen(x);
      const _y = context.gen(y);
      const shapeX = getShape(_x); //.shape;
      const shapeY = getShape(_y); //.shape;
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
        const vectorSize = shapeY[0];
        code = `
// broadcast
let batchIndex = index / ${vectorSize}u;
let vectorIndex = index % ${vectorSize}u;
let ${variableName} = ${toScalar(_x)} ${op} ${toScalar(_y, "vectorIndex")};
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
  ) => BackPropagationOutput,
) => {
  let code = "";
  let intermediateVariables: string[] = [];
  let gradientOutputs: string[] = [];
  const visited = new Set<string>();
  const shapes = node.dependencies.map((dep) => getShape(dep));
  const isBroadcasting =
    shapes[0].length !== shapes[1].length || !arraysEqual(shapes[0], shapes[1]);

  console.log("calculating gradients for add", node.variable, node);

  for (let i = node.dependencies.length - 1; i >= 0; i--) {
    console.log("LOOP i=%s", i);
    const dep = node.dependencies[i];
    if (visited.has(dep.gradientVariable)) {
      console.log("already visited...");
      continue;
    }
    visited.add(dep.gradientVariable);

    const grad = customLogic
      ? customLogic(dep, i, isBroadcasting, shapes)
      : { code: `${dep.gradientVariable} += ${gradOut};\n` };

    console.log("grad we got", grad, grad.gradientOutputs);
    code += grad.code;
    if ((grad as BackPropagationOutput).intermediateVariables) {
      intermediateVariables.push(...(grad as BackPropagationOutput).intermediateVariables);
    }
    if ((grad as BackPropagationOutput).gradientOutputs) {
      console.log("adding to gradient for add", grad);
      gradientOutputs.push(...(grad as BackPropagationOutput).gradientOutputs);
    }
  }

  console.log("gradient outputs for add=", gradientOutputs);
  return {
    code,
    intermediateVariables: intermediateVariables.map((x) => trimIndex(x)),
    gradientOutputs,
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
        code: `${dep.gradientVariable} += ${gradOut};\n // add grad`,
        intermediateVariables: [gradOut],
        gradientOutputs: [dep.gradientVariable],
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
            code: `${dep.gradientVariable} += ${gradOut};\n // matrix i==0`,
            intermediateVariables: [gradOut],
            gradientOutputs: [dep.gradientVariable, gradOut],
          };
        } else {
          // Vector (bias)
          let intermediate = `grad_${node.parent?.variable}_output`;
          return {
            code: `
              // add grad broadcast
              let vectorIndex = index % ${vectorSize}u;
              var grad_sum = 0.0;
              for (var i = 0u; i < ${batchSize}u; i = i + 1u) {
                grad_sum += ${intermediate}[i * ${vectorSize}u + vectorIndex];
              }
              ${dep.gradientVariable} = grad_sum;
            `,
            intermediateVariables: [gradOut],
            gradientOutputs: [dep.gradientVariable],
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

/**
 * Converts an ASTNode to the code representation needed in a backpropagation kernel
 * */
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
        gradientOutputs: [dep.gradientVariable],
      };
    } else {
      const code = `${dep.gradientVariable} += ${gradOut} * ${v(otherDep)};\n`;
      return {
        code,
        intermediateVariables: [gradOut, v(otherDep)],
        gradientOutputs: i === 0 ? [dep.gradientVariable] : [dep.gradientVariable],
      };
    }
  });
});

export const div = binaryOp("div", "/", (node: ASTNode, gradOut: string) =>
  grad(node, gradOut, (dep, i) => {
    if (i === 0) {
      // Gradient for the first operand (dividend)
      const code = `${dep.gradientVariable} += ${gradOut} / ${v(node.dependencies[1])}; // divide A \n`;
      return {
        code,
        intermediateVariables: [gradOut, v(node.dependencies[i])],
        gradientOutputs: [dep.gradientVariable],
      };
    }
    // Gradient for the second operand (divisor)
    const code = `${dep.gradientVariable} += -${gradOut} * ${v(node.dependencies[0])} / (${v(dep)} * ${v(dep)}); // divide B \n`;
    return {
      code,
      intermediateVariables: [gradOut, v(node.dependencies[0]), v(dep)],
      gradientOutputs: [dep.gradientVariable],
    };
  }),
);

export const func = (
  name: string,
  forward: (x: string) => string,
  derivative: (x: string) => string,
) => {
  return (freq: Arg) => {
    return memo(
      (context: Context<ASTNode>): ASTNode => {
        context = context.useContext(OpType.Regular);
        const [variableName] = context.useVariables(`${name}_result`);
        const _freq = context.gen(freq);
        const code = `
let ${variableName} = ${forward(toScalar(_freq))};
  `;
        return context.emit(name, variableName, code, OpType.Regular, getShape(_freq), _freq);
      },
      (node: ASTNode, gradOut: string) => {
        const inputVar = node.dependencies[0].variable;
        const gradientCode = `
let grad_${inputVar} = ${gradOut} * ${derivative(toScalar(node.dependencies[0], undefined, true))};
        `;
        return {
          code: gradientCode,
          intermediateVariables: emitIntermediate(node),
          gradientOutputs: [`grad_${inputVar}`],
        };
      },
      freq,
    );
  };
};

export const sqrt = func(
  "sqrt",
  (x: string) => `sqrt(${x})`,
  (x: string) => `0.5 / sqrt(${x})`,
);
export const pow2 = func(
  "pow2",
  (x: string) => `pow(${x}, 2)`,
  (x: string) => `2.0 * ${x}`,
); // For x^2
export const pow3 = func(
  "pow3",
  (x: string) => `pow(${x}, 3)`,
  (x: string) => `3.0 * pow(${x}, 2)`,
); // For x^3
