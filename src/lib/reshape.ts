import { Arg, OpType, ASTNode, DataType } from "./zen";
import { Context } from "./context";

export const reshape =
  (input: Arg, newShape: number[]) =>
  (context: Context): ASTNode => {
    const _input = context.gen(input, true);
    const ret = {
      ..._input,
      shape: newShape,
      opType: OpType.Reshape,
    };
    return ret;
 };
