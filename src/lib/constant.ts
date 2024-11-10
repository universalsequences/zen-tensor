import { Context } from "./context";
import { ASTNode, NodeGen, OpType } from "./zen";

export const constant = (shape: [number, number], value: number) => {
  let memoized: NodeGen | undefined;
  return (context: Context<ASTNode>) => {
    if (memoized) {
      return memoized(context);
    }
    const tensor = context.tensorGraph.tensor(shape, "const").fill(value);
    memoized = tensor.gen();
    return memoized(context);
  };
};
