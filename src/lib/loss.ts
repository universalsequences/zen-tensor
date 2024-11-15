import { Context } from "./context";
import { trimIndex, v } from "./math";
import { memo } from "./memo";
import { emitIntermediate } from "./utils";
import { Arg, ASTNode, intermediate, OpType } from "./zen";

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
      let predictedVar = node.dependencies[0]; // Variable for predicted output from forward pass
      const actualVar = node.dependencies[1].variable; // Variable for actual labels from forward pass
      const clampedPredictedVar = `clamp(${intermediate(predictedVar)}[index], 1e-7, 1.0 - 1e-7)`;

      // Gradient with respect to the predicted output
      const code = `
let y_pred = ${clampedPredictedVar};
let y_true = ${actualVar}[index];
${node.gradientVariable} = (y_pred - y_true) / (y_pred * (1.0 - y_pred));
`;
      // Return the generated code and any intermediate variables needed for further operations
      return {
        code: code.trim(),
        intermediateVariables: [
          trimIndex(v(node.dependencies[0])),
          trimIndex(v(node.dependencies[1])),
        ],
        gradientOutputs: [node.gradientVariable],
      };
    },
    predicted,
    actual,
  );

export const meanSquaredError = (predictions: Arg, targets: Arg) =>
  memo(
    (context: Context<ASTNode>): ASTNode => {
      context = context.useContext(OpType.Reduction);
      const _pred = context.gen(predictions);
      const _targets = context.gen(targets);
      const [result] = context.useVariables("mse_result");

      const code = `
        let diff = ${v(_pred)} - ${v(_targets)};
        let ${result} = diff * diff;
      `;

      return context.emit("mse", result, code, OpType.Reduction, _pred.shape, _pred, _targets);
    },
    (node: ASTNode, gradOut: string) => {
      const gradCode = `
        ${node.gradientVariable} = 2.0 * (${v(node.dependencies[0])} - ${v(node.dependencies[1])}) * ${gradOut};
      `;
      return {
        code: gradCode,
        intermediateVariables: [
          trimIndex(v(node.dependencies[0])),
          trimIndex(v(node.dependencies[1])),
        ],
        gradientOutputs: [node.gradientVariable],
      };
    },
    predictions,
    targets,
  );

/*
export const crossEntropy = (predictions: Arg, targets: Arg) =>
  memo(
    (_context: Context<ASTNode>): ASTNode => {
      const context = _context.useContext(OpType.Reduction);
      const _pred = context.gen(predictions);
      const _targets = context.gen(targets);
      const [result] = context.useVariables("cross_entropy_result");

      // Get batch size and vocab size from shape
      const batchSize = _pred.shape[0];
      const vocabSize = _pred.shape[1];

      const code = `
          // Cross entropy: -sum(target * log(pred)) per batch item
          let batchIndex = index / ${vocabSize}u;
          let inBatch = batchIndex < ${batchSize}u;
          var ${result} = 0.0;
          if (inBatch) {
            ${result} = -${v(_targets)} * log(${v(_pred)} + 1e-7);
            // Scale by batch size to normalize gradient
            ${result} = ${result}; // ${batchSize}.0;
          }
        `;

      return context.emit(
        "cross_entropy",
        result,
        code,
        OpType.Reduction,
        _pred.shape,
        _pred,
        _targets,
      );
    },
    (node: ASTNode, gradOut: string) => {
      // Gradient of cross entropy with respect to predictions
      // Scale gradient by batch size for proper normalization
      const batchSize = node.shape[0];
      const vocabSize = node.shape[1];

      const gradCode = `
          // cross entropy gradient ${node.variable}
          let batchIndex = index / ${vocabSize}u;
          let inBatch = batchIndex < ${batchSize}u;
          if (inBatch) {
            ${node.gradientVariable} = ${gradOut} * (${v(node.dependencies[0])} - ${v(node.dependencies[1])});
            // Scale by batch size to normalize gradient
            ${node.gradientVariable} = ${node.gradientVariable}; // ${batchSize}.0;
          }
        `;

      return {
        code: gradCode,
        intermediateVariables: emitIntermediate(node),
        gradientOutputs: [node.gradientVariable],
      };
    },
    predictions,
    targets,
  );
  */

export const crossEntropy = (predictions: Arg, targets: Arg) =>
  memo(
    (_context: Context<ASTNode>): ASTNode => {
      const context = _context.useContext(OpType.Reduction);
      const _pred = context.gen(predictions);
      const _targets = context.gen(targets);
      const [result] = context.useVariables("cross_entropy_result");
      const code = `
          let ${result} = -${v(_targets)} * log(${v(_pred)} + 1e-7);
        `;
      return context.emit(
        "cross_entropy",
        result,
        code,
        OpType.Reduction,
        _pred.shape,
        _pred,
        _targets,
      );
    },
    (node: ASTNode, gradOut: string) => {
      const gradCode = `
          // cross entropy gradient ${node.variable}
          ${node.gradientVariable} = ${gradOut} * (${v(node.dependencies[0])} - ${v(node.dependencies[1])});
        `;
      return {
        code: gradCode,
        intermediateVariables: emitIntermediate(node),
        gradientOutputs: [node.gradientVariable],
      };
    },
    predictions,
    targets,
  );
