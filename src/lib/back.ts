import { v } from "./math";
import { constructGroup } from "./utils";
import { ASTNode, intermediate, intermediateVar } from "./zen";

export type BGen = (node: ASTNode, x: string) => string;

// everything needed to get the kernel for this
export interface BackwardContext {
  code: string;
  inputs: string[];
  outputs: string[];
}

export const backpass = (finalNode: ASTNode, gradInit = "1.0"): BackwardContext[] => {
  console.log("backpass called with gradInit=", gradInit, finalNode);
  let otherKernels: BackwardContext[] = [];
  let backwardCode = "";
  let outputCode = "";
  let initializations = "";
  const gradientInitializations = new Set<string>();
  const visited = new Set<ASTNode>();
  const inputNodes = new Set<ASTNode>();
  const outputs: string[] = [];
  const inputs: string[] = [];

  // Recursive function to generate backward code, processing the current node first
  const generateBackwardCode = (node: ASTNode, gradOut: string): void => {
    // Skip if node has already been visited
    if (visited.has(node)) return;
    visited.add(node);
    if (node.context !== finalNode.context) {
      otherKernels.push(...backpass(node, gradOut));
      return;
    }

    // Ensure gradient variables are initialized once before usage
    if (node.gradientVariable && !gradientInitializations.has(node.gradientVariable)) {
      // Initialize the final output gradient to 1.0, others to 0.0
      const initValue = node === finalNode ? gradInit : "0.0";
      initializations += `var ${node.gradientVariable} = ${initValue}; // initializer \n`;
      gradientInitializations.add(node.gradientVariable);
    }

    // Generate the backpropagation code for the current node
    if (node.backprop) {
      const backpropCode = node.backprop(gradOut);
      if (backpropCode.includes(intermediate(node))) {
        inputs.push(intermediate(node));
      }
      for (const inp of finalNode.context.getInputs()) {
        if (node.dependencies.some((x) => x.variable === inp)) {
          inputs.push(inp);
        }
      }

      if (backpropCode !== "") {
        backwardCode += backpropCode + "\n";
      }
    }

    // Process dependencies (children) after processing the current node
    node.dependencies.forEach((dep) => {
      const depGradOut = dep.gradientVariable ? v(dep) : `grad_${dep.variable}`;
      generateBackwardCode(dep, depGradOut);
    });

    // Identify input nodes (leaf nodes with no dependencies)
    if (node.dependencies.length === 0) {
      inputNodes.add(node);
    }
  };

  // Start with the root of the AST (final operation)
  generateBackwardCode(finalNode, finalNode.gradientVariable);

  // Generate output code for leaf nodes (inputs)
  const visitedInputs = new Set<string>();
  inputNodes.forEach((node) => {
    if (!visitedInputs.has(node.variable)) {
      visitedInputs.add(node.variable);
      const output = `grad_${node.variable}_output`;
      outputs.push(output);
      outputCode += `      ${output}[index] = ${node.gradientVariable};\n`;
    }
  });

  const inputsCode = inputs.map((input, index) => constructGroup(index, "read", input)).join("\n");
  const outputsCode = outputs
    .map((output, index) => constructGroup(inputs.length + index, "read_write", output))
    .join("\n");

  // Prepend initializations to the backward code
  const kernel = `
${inputsCode}
${outputsCode}
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let index = global_id.x;
${trim(initializations)}
${trim(backwardCode)}
      // Output gradients for input tensors
${trim(outputCode)}
    }
  `;

  // TODO - each context shall have a BackwardContext that contains everything needed to run
  // the backwards version of the context (including the exact intermediate values/buffers needed)

  finalNode.context.backward = {
    code: kernel,
    outputs,
    inputs,
  };
  return [finalNode.context.backward, ...otherKernels];
};

const trim = (x: string) =>
  x
    .split("\n")
    .map((x) => x.trim())
    .filter((x) => x !== "")
    .map((x) => `      ${x}`)
    .join("\n");
