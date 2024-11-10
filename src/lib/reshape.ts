import { Arg, OpType, ASTNode } from "./zen";
import { Context } from "./context";

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
  return (context: Context<ASTNode>): ASTNode => {
    const _input = context.gen(input, true);
    return {
      ..._input,
      shape: [input.shape[1], input.shape[0]],
      transposed: _input.transposed ? false : true,
      opType: OpType.Reshape,
    };
  };
};

/**
 * buffer: [0, 0, 0, 1, 1, 1] (w/ shape [3,2]) accessed originally as [0,0,0],[1,1,1]
 * transposed: [[0, 0, 0], [1, 1, 1]] (shape: [3,2])-> [[0, 1], [0, 1], [0,1]] (shape: [2,3])
 *
 * */
export const accessIndex = (node: ASTNode, row: string, col: string) => {
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
