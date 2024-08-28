import { Context } from "./context";
import { Arg, ASTNode, PartialGen } from "./zen";

// TODO - add backward PartialGen representing the backpropagation functions for calculating gradients
export const memo = (forward: PartialGen, ...args: Arg[]) => {
  let memoized: ASTNode | undefined = undefined;

  return (context: Context): ASTNode => {
    if (memoized) {
      return memoized;
    }
    memoized = forward(context, ...args);
    return memoized;
  };
};
