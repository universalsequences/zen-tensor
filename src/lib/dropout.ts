import { v } from "./math";
import { Arg, ASTNode } from "./zen";
import { OpType, matmul, div, Context, sub, add, mult, transpose, getShape } from "./index";
import { constant } from "./constant";
import { memo } from "./memo";
import { emitIntermediate } from "./utils";

export const dropout = (x: Arg, rate: number) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      context = context.useContext(OpType.Regular);
      const _x = context.gen(x);
      const [result] = context.useVariables("dropout_result");

      const scale = 1 / (1 - rate);

      const code = `
        let ${result} = select(0.0, ${v(_x)} * ${scale}, random() > ${rate});
      `;

      return context.emit("dropout", result, code, OpType.Regular, _x.shape, _x);
    },
    (node: ASTNode, gradOut: string) => {
      const gradCode = `
        ${node.gradientVariable} = select(0.0, ${gradOut} * ${1 / (1 - rate)}, random() > ${rate});
      `;
      return {
        code: gradCode,
        intermediateVariables: emitIntermediate(node),
      };
    },
    x,
  );
