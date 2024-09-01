import { Context } from "./context";
import { ASTNode, OpType } from "./zen";

export const numberOp = (num: number) => {
  return (context: Context<ASTNode>) => {
    let [numVariable] = context.useVariables("number");
    let code = `let ${numVariable}: f32 = ${num};`;
    return context.emit("number", numVariable, code, OpType.Regular, [1]);
  };
};
