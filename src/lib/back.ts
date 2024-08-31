import { trimIndex, v } from "./math";
import { constructGroup } from "./utils";
import { ASTNode, intermediate, intermediateVar } from "./zen";

export type BGen = (
  node: ASTNode,
  x: string,
) => {
  code: string;
  intermediateVariables: string[];
};

// everything needed to get the kernel for this
export interface BackwardContext {
  code: string;
  inputs: string[];
  outputs: string[];
}

export const backpass = (finalNode: ASTNode, gradInit = "1.0"): BackwardContext[] => {
  console.log("BACKPASS finalNode.context.id=", finalNode.context.id);
  let otherKernels: BackwardContext[] = [];
  let backwardCode = "";
  let outputCode = "";
  let initializations = "";
  const gradientInitializations = new Set<string>();
  const visited = new Set<ASTNode>();
  const inputNodes = new Set<ASTNode>();
  const crossNodes = new Set<ASTNode>();
  const outputs: string[] = [];
  let inputs: string[] = [];
  const saved = new Set<string>();

  // Recursive function to generate backward code, processing the current node first
  const generateBackwardCode = (node: ASTNode, gradOut: string): void => {
    // Skip if node has already been visited
    if (visited.has(node)) return;
    visited.add(node);
    if (node.context !== finalNode.context) {
      if (node.variable.includes("cross")) {
        crossNodes.add(node);
      }
      otherKernels.push(...backpass(node, gradOut));
      return;
    }

    // Ensure gradient variables are initialized once before usage
    if (node.gradientVariable && !gradientInitializations.has(node.gradientVariable)) {
      // Initialize the final output gradient to 1.0, others to 0.0
      const initValue = node === finalNode ? gradInit : "0.0";
      initializations += `var ${node.gradientVariable} = ${initValue}; // initializer \n`;
      gradientInitializations.add(node.gradientVariable);
      saved.add(node.gradientVariable);
      if (initValue !== "1.0" && initValue !== "0.0") {
        console.log("adding finalNode.variable because init was ...", finalNode.variable);
        // inputs.push(finalNode.variable);
      }
    }
    for (const dep of node.dependencies) {
      gradientInitializations.add(dep.gradientVariable);
    }

    // Generate the backpropagation code for the current node
    if (node.backprop) {
      const re = node.backprop(gradOut);
      if (
        re.code.includes("grad_tensor_0 += grad_mult_result1 * add_result1_intermediate[index];")
      ) {
      }
      const { code: backpropCode, intermediateVariables } = re;
      if (intermediateVariables) {
        console.log("adding intermediates", intermediateVariables);
        inputs.push(...intermediateVariables);
      }
      if (backpropCode.includes(intermediate(node))) {
        console.log("adding intermediates 2", intermediate(node));
        inputs.push(intermediate(node));
      }
      for (const inp of finalNode.context.getInputs()) {
        if (node.dependencies.some((x) => x.variable === inp)) {
          console.log("adding getInput dep", inp);
          // TODO - is this necessary?
          // inputs.push(inp);
        }
      }

      if (backpropCode !== "") {
        backwardCode += backpropCode + "\n";
      }
    }

    // Process dependencies (children) after processing the current node
    node.dependencies.forEach((dep) => {
      const depGradOut = dep.gradientVariable ? v(dep) : `grad_${dep.variable}`;
      inputNodes.add(node);
      console.log("recursive generate", depGradOut, dep, node);

      const output = `grad_${node.variable}_output`;
      generateBackwardCode(dep, `${output}[index]`);
    });

    // Identify input nodes (leaf nodes with no dependencies)
    if (node.dependencies.length === 0) {
      inputNodes.add(node);
    }
  };

  // Start with the root of the AST (final operation)
  generateBackwardCode(finalNode, finalNode.gradientVariable);
  inputs = Array.from(new Set(inputs));

  if (
    gradInit !== "1.0" &&
    (initializations.includes(trimIndex(gradInit)) || backwardCode.includes(trimIndex(gradInit)))
  ) {
    if (!initializations.includes(trimIndex(gradInit) + "=")) {
      console.log("trim addinging gradInit2=", gradInit);
      inputs.push(trimIndex(gradInit));
    }
  }

  for (const init of gradientInitializations) {
    if (!saved.has(init)) initializations += `var ${init} = 0.0;\n`;
    if (inputs.includes(init)) {
      inputs = inputs.filter((x) => x !== init);
    }
  }

  console.log("finished finalNode.context", finalNode.context.id);
  console.log("saveds=", saved);
  console.log("inputs=", inputs);
  console.log("inputNodes=", inputNodes);

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

  crossNodes.forEach((node) => {
    if (!visitedInputs.has(node.variable)) {
      visitedInputs.add(node.variable);
      const output = node.variable;
      outputs.push(output);
      outputCode += `      ${output}[index] = ${node.gradientVariable};\n`;
    }
  });

  console.log("inputs code inputs=", inputs);
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
