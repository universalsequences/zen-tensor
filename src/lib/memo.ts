import { BGen } from "./back";
import { Context } from "./context";
import { Arg, ASTNode, PartialGen } from "./zen";

// TODO - add backward PartialGen representing the backpropagation functions for calculating gradients
export const memo = (forward: PartialGen, backward: BGen, ...args: Arg[]) => {
  let memoized: ASTNode | undefined = undefined;

  return (context: Context<ASTNode>): ASTNode => {
    if (memoized) {
      return memoized;
    }
    const node = forward(context, ...args);
    node.backprop = (gradOut: string) => {
      return backward(node, gradOut);
    };
    memoized = node;
    return node;
  };
};
