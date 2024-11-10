import { Context } from "./context";
import { ASTNode } from "./zen";

export const constant = (shape: [number, number], value: number) => {
  let memoized: ASTNode | undefined = undefined;
  return (context: Context<ASTNode>) => {
    const [v] = context.useVariables("constant");
    const tensor = context.tensorGraph.tensor(shape, v).fill(value);
    if (memoized) {
      return memoized;
    }
    memoized = tensor.gen()(context);
    return memoized;
  };
};
