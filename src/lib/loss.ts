import { Context } from "./context";
import { trimIndex, v } from "./math";
import { memo } from "./memo";
import { Arg, ASTNode, OpType } from "./zen";

const epsilon = 1e-7; // Small value to prevent log(0)

export const binaryCrossEntropy = (predicted: Arg, actual: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables(`bce_result`);

      const _predicted = context.gen(predicted);
      const _actual = context.gen(actual);

      // Clamp the predicted values to prevent log(0) or log(1)
      const clampedPredicted = `clamp(${v(_predicted)}, ${epsilon}, 1.0 - ${epsilon})`;

      // -y * log(a2) - (1 - y) * log(1 - a2)
      let code = `
        let ${res} = -(${v(_actual)} * log(${clampedPredicted}) +
        (1.0 - ${v(_actual)}) * log(1.0 - ${clampedPredicted}));
      `;
      return context.emit(
        "binaryCrossEntropy",
        res,
        code,
        OpType.Regular,
        _predicted.shape,
        _predicted,
        _actual,
      );
    },
    (node: ASTNode, gradOut: string) => {
      const predictedVar = node.dependencies[0].variable; // Variable for predicted output from forward pass
      const actualVar = node.dependencies[1].variable; // Variable for actual labels from forward pass
      const clampedPredictedVar = `clamp(${predictedVar}[index], 1e-7, 1.0 - 1e-7)`;

      // Gradient with respect to the predicted output
      const gradPredictedCode = `
${node.gradientVariable} = (y_pred - y_true) / (y_pred * (1.0 - y_pred));
`;

      // We'll remove the gradient calculation for actual labels as it's not needed for backpropagation
      const gradActualCode = "";

      // Combine both gradient codes into a single code block
      const code = `
let y_pred = ${clampedPredictedVar};
let y_true = ${actualVar}[index];
${gradPredictedCode}
${gradActualCode}
`;
      // Return the generated code and any intermediate variables needed for further operations
      return {
        code: code.trim(),
        intermediateVariables: [
          trimIndex(v(node.dependencies[0])),
          trimIndex(v(node.dependencies[1])),
        ],
      };
    },
    predicted,
    actual,
  );
