import { Context } from "./context";
import { Tensor } from "./input";

export enum OpType {
  Regular = 0,
  Reduction = 1,
  Reshape = 2,
}

export type Gen = (context: Context<ASTNode>) => ASTNode;

export type Arg = Gen | Tensor | number;

export type PartialGen = (context: Context<ASTNode>, ...arg: Arg[]) => ASTNode;

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
  context: Context<ASTNode>;
  type: DataType;
  shape: number[]; // [rows, cols] for 2D, [length] for 1D
  backprop?: (x: string) => {gg
    code: string;
    intermediateVariables: string[];
  };
  gradientVariable: string; // New field for gradient variable
  parent?: ASTNode;
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

export const intermediate = (a: ASTNode) => intermediateVar(a.variable);
export const intermediateVar = (a: string) => `${a}_intermediate`;
