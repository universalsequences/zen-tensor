import type { BGen } from "./back";
import type { Context } from "./context";
import type { Arg, ASTNode, PartialGen, NodeGen } from "./zen";

export const memo = (forward: PartialGen, backward: BGen, ...args: Arg[]) => {
  let memoized: ASTNode | undefined = undefined;
  const resultGen = (context: Context<ASTNode>): ASTNode => {
    if (memoized) {
      const variable = memoized.variable.slice(
        0,
        memoized.variable.length - "_intermediate".length,
      );
      if (!context.lazyInputs.includes(variable)) {
        context.lazyInputs.push(variable);
      }
      context.lazyInputShapes.set(variable, memoized.shape);
      return memoized;
    }
    const node = forward(context, ...args);

    node.backprop = (node: ASTNode, gradOut: string) => {
      // store backward propagation
      return backward(node, gradOut);
    };
    memoized = node;
    (resultGen as NodeGen).node = node;
    return node;
  };
  return resultGen as NodeGen;
};
