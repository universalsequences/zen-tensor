import { Context } from "./context";
import { trimIndex, v } from "./math";
import { memo } from "./memo";
import { Arg, ASTNode, OpType } from "./zen";

export const relu = (x: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables(`relu_result`);

      const _x = context.gen(x);

      let code = `let ${res} = max(0.0, ${v(_x)});`;
      return context.emit("relu", res, code, OpType.Regular, _x.shape, _x);
    },
    (node: ASTNode, gradOut: string) => {
      const inputVar = node.dependencies[0].variable;
      const gradCode = `
        let grad_${inputVar} = select(0.0, ${gradOut}, ${v(node.dependencies[0])} > 0.0);
      `;
      return {
        code: gradCode,
        intermediateVariables: [trimIndex(v(node.dependencies[0]))],
      };
    },
    x,
  );

export const sigmoid = (x: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables(`sigmoid_result`);
      const _x = context.gen(x);
      let code = `let ${res} = 1.0 / (1.0 + exp(-${v(_x)}));`;
      return context.emit("sigmoid", res, code, OpType.Regular, _x.shape, _x);
    },
    (node: ASTNode, gradOut: string) => {
      const inputVar = node.dependencies[0].variable;
      const gradCode = `
        let sigmoid_${inputVar} = 1.0 / (1.0 + exp(-${v(node.dependencies[0])}));
        let grad_${inputVar} = ${gradOut} * sigmoid_${inputVar} * (1.0 - sigmoid_${inputVar});
      `;
      return {
        code: gradCode,
        intermediateVariables: [trimIndex(v(node.dependencies[0]))],
      };
    },
    x,
  );
