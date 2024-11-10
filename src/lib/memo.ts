import { BGen } from "./back";
import { Context } from "./context";
import { Arg, ASTNode, PartialGen, NodeGen } from "./zen";

// TODO - add backward PartialGen representing the backpropagation functions for calculating gradients
let i = 0;
export const memo = (forward: PartialGen, backward: BGen, ...args: Arg[]) => {
  let memoized: ASTNode | undefined = undefined;
  const id = i++;

  const resultGen = (context: Context<ASTNode>): ASTNode => {
    if (memoized) {
      const variable = memoized.variable.slice(
        0,
        memoized.variable.length - "_intermediate".length,
      );
      context.lazyInputs.push(variable);
      context.lazyInputShapes.set(variable, memoized.shape);
      console.log("returning memoized id=%s", i, memoized);
      return memoized;
    }
    const node = forward(context, ...args);
    node.backprop = (gradOut: string) => {
      return backward(node, gradOut);
    };
    memoized = node;
    (resultGen as NodeGen).node = node;
    console.log("saving memoized id=%s", id, node.operation, node);
    return node;
  };
  return resultGen as NodeGen;
};
