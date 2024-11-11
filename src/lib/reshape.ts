import { Arg, OpType, ASTNode, NodeGen } from "./zen";
import { Context } from "./context";
import { memo } from "./memo";

export const reshape =
  (input: Arg, newShape: number[]) =>
  (context: Context<ASTNode>): ASTNode => {
    return {
      ...context.gen(input, true),
      shape: newShape,
      opType: OpType.Reshape,
    };
  };

export const transpose = (input: Arg) => {
  const nodeGen = (context: Context<ASTNode>): ASTNode => {
    const _input = context.gen(input);
    console.log("transposing input =", _input.variable, _input, context);
    if (_input.variable.endsWith("_intermediate")) {
      const variable = _input.variable.slice(0, _input.variable.length - "_intermediate".length);
      context.lazyInputs.push(variable);
      context.lazyInputShapes.set(variable, _input.shape);
    }
    const node = {
      ..._input,
      transposed: true,
      opType: OpType.Regular,
    };
    (nodeGen as NodeGen).node = node;
    return node;
  };
  return nodeGen;
};

/**
 * buffer: [0, 0, 0, 1, 1, 1] (w/ shape [3,2]) accessed originally as [0,0,0],[1,1,1]
 * transposed: [[0, 0, 0], [1, 1, 1]] (shape: [3,2])-> [[0, 1], [0, 1], [0,1]] (shape: [2,3])
 *
 * */
export const getIndex = (node: ASTNode, row: string, col: string) => {
  const [rows, cols] = getShape(node);
  if (node.transposed) {
    return `${col} * ${rows} + ${row}`;
  } else {
    return `${row} * ${cols} + ${col}`;
  }
};

export const getShape = (node: ASTNode) => {
  return node.transposed ? [...node.shape].reverse() : node.shape;
};
