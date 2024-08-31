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
      console.log("relu for=", _x);

      let code = `let ${res} = max(0.0, ${v(_x)});`;
      return context.emit(res, code, OpType.Regular, _x.shape, _x);
    },
    (node: ASTNode, gradOut: string) => {
      console.log("relu back=", node, gradOut);
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
