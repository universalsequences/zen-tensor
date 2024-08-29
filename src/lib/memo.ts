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
      let code = `
        ${backward(node, gradOut)}
`;
      for (const dep of node.dependencies) {
        code += `${dep.gradientVariable} += grad_${dep.variable};

`;
      }
      return code;
  };
  memoized = node;
  return node;
};
}
