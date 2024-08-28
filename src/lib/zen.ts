import { Context } from "./context";
import { Tensor } from "./input";

export enum OpType {
  Regular = 0,
  Reduction = 1,
  Reshape = 2,
}

export type Gen = (context: Context) => ASTNode;

export type Arg = Gen | Tensor | number;

export type PartialGen = (context: Context, ...arg: Arg[]) => ASTNode;

export enum DataType {
  Scalar = 0,
  Tensor = 1,
}

/**
 * Each operation returns GenResult, effectively as an node in the AST
 * */
export interface ASTNode {
  variable: string;
  code: string; // the code contribution to this kernel (for this AST node)
  dependencies: ASTNode[];
  opType: OpType;
  context: Context;
  type: DataType;
  shape: number[]; // [rows, cols] for 2D, [length] for 1D
}

/**
 * Used to convert a piece of data into a scalar representation (as a codegen string)
 *
 * */
export const toScalar = (data: ASTNode, type: DataType = DataType.Scalar, index?: string) => {
  if (data.type === DataType.Tensor && type === DataType.Scalar) {
    if (index) {
      return `${data.variable}[${index}]`;
    } else {
      return `${data.variable}[index]`;
    }
  }
  return data.variable;
};
