import { BGen } from "./back";
import { Context } from "./context";
import { Arg, ASTNode, PartialGen, NodeGen } from "./zen";

// TODO - add backward PartialGen representing the backpropagation functions for calculating gradients
export const memo = (forward: PartialGen, backward: BGen, ...args: Arg[]) => {
  let memoized: ASTNode | undefined = undefined;

  const resultGen = (context: Context<ASTNode>): ASTNode => {
    if (memoized) {
      return memoized;
    }
    const node = forward(context, ...args);
    node.backprop = (gradOut: string) => {
      return backward(node, gradOut);
    };
    memoized = node;
    (resultGen as NodeGen).node = node;
    return node;
  };
  return resultGen as NodeGen;
};
