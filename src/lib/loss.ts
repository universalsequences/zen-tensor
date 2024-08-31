import { Context } from "./context";
import { trimIndex, v } from "./math";
import { memo } from "./memo";
import { Arg, ASTNode, OpType } from "./zen";

/*
export const binaryCrossEntropy = (predicted: Arg, actual: Arg) =>
  memo(
    (c: Context<ASTNode>) => {
      const context = c.useContext(OpType.Regular);
      const [res] = context.useVariables(`bce_result`);

      const _predicted = context.gen(predicted);
      const _actual = context.gen(actual);

      // -y * log(a2) - (1 - y) * log(1 - a2)
      let code = `let ${res} = -(${v(_actual)} * log(${v(_predicted)}) +
(1.0 - ${v(_actual)}) * log(1.0 - ${v(_predicted)}));`;
      return context.emit(res, code, OpType.Regular, _predicted.shape, _predicted, _actual);
    },
    (node: ASTNode) => {
      const predictedVar = node.dependencies[0].gradientVariable;
      const actualVar = node.dependencies[1].gradientVariable;

      const gradPredictedCode = `
        ${predictedVar} += (${v(node.dependencies[0])} - ${v(node.dependencies[1])}) / (${v(node.dependencies[0])} * (1.0 - ${v(node.dependencies[0])}));
      `;

      const gradActualCode = `
        ${actualVar} += -log(${v(node.dependencies[0])} / (1.0 - ${v(node.dependencies[0])}));
      `;

      return {
        code: gradPredictedCode + gradActualCode,
        intermediateVariables: [
          trimIndex(v(node.dependencies[0])),
          trimIndex(v(node.dependencies[1])),
        ],
      };
    },
    predicted,
    actual,
  );

  */

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
      return context.emit(res, code, OpType.Regular, _predicted.shape, _predicted, _actual);
    },
    (node: ASTNode, gradOut: string) => {
      const predictedVar = node.dependencies[0].variable; // Variable for predicted output from forward pass
      const actualVar = node.dependencies[1].variable; // Variable for actual labels from forward pass

      const clampedPredictedVar = `clamp(${predictedVar}[index], 1e-7, 1.0 - 1e-7)`;

      // Gradient with respect to the predicted output
      const gradPredictedCode = `
    ${node.gradientVariable} = (${clampedPredictedVar} - ${v(node.dependencies[1])}) /
    (${clampedPredictedVar} * (1.0 - ${clampedPredictedVar}));
  `;

      // Gradient with respect to the actual labels (usually not used, but included for completeness)
      const gradActualCode = `
    ${node.dependencies[1].gradientVariable} +=
    -${v(node.dependencies[1])} / ${clampedPredictedVar} +
    (1.0 - ${v(node.dependencies[1])}) / (1.0 - ${clampedPredictedVar});
  `;

      // Combine both gradient codes into a single code block
      const code = `
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
