import { trimIndex, v } from "./math";
import { getShape } from "./reshape";
import { constructGroup, shapeToSize } from "./utils";
import { type ASTNode, type BackPropagationOutput, intermediate } from "./zen";

export type BGen = (node: ASTNode, x: string) => BackPropagationOutput;

// everything needed to get the kernel for this
export interface BackwardContext {
  code: string;
  inputs: string[];
  outputs: string[];
}

export const backpass = (
  finalNode: ASTNode,
  gradInit = "1.0",
  gradientsWritten = new Map<string, number>(),
): BackwardContext[] => {
  const gradientsWrittenLocal = new Map<string, number>();
  for (const [key, value] of gradientsWritten) {
    gradientsWrittenLocal.set(key, value);
  }
  console.log("backpass finalNode=", finalNode.variable, gradInit);
  const otherKernels: BackwardContext[] = [];
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
  const gradientOutputs: string[] = [];

  // Recursive function to generate backward code, processing the current node first
  const generateBackwardCode = (node: ASTNode, gradOut: string): void => {
    // Skip if node has already been visited
    if (visited.has(node)) return;
    visited.add(node);
    if (node.context !== finalNode.context) {
      // context switched or the inputs have grown more than 4  and we should partition the kernel
      if (node.variable.includes("cross")) {
        crossNodes.add(node);
      }
      otherKernels.push(...backpass(node, gradOut, gradientsWritten));
      return;
    }

    // Ensure gradient variables are initialized once before usage
    if (node.gradientVariable && !gradientInitializations.has(node.gradientVariable)) {
      // Initialize the final output gradient to 1.0, others to 0.0
      const initValue = gradInit; ///*node === finalNode ? gradInit : "0.0";
      initializations += `var ${node.gradientVariable} = ${initValue}; // initializer \n`;
      gradientInitializations.add(node.gradientVariable);
      saved.add(node.gradientVariable);
    }
    for (const dep of node.dependencies) {
      gradientInitializations.add(dep.gradientVariable);
    }

    // Generate the backpropagation code for the current node
    if (node.backprop) {
      const re = node.backprop(node, gradOut);
      const { code: backpropCode, intermediateVariables } = re;
      if (re.gradientOutputs) {
        for (const gradientOutput of re.gradientOutputs) {
          if (!gradientsWritten.has(gradientOutput)) {
            gradientsWritten.set(gradientOutput, 1);
            gradientsWrittenLocal.set(gradientOutput, 1);
            console.log("setting gradientOutput=", gradientOutput, 1);
          } else {
            gradientsWritten.set(gradientOutput, (gradientsWritten.get(gradientOutput) ?? 0) + 1);
            gradientsWrittenLocal.set(
              gradientOutput,
              gradientsWrittenLocal.get(gradientOutput) + 1,
            );
            console.log(
              "setting gradientOutput=",
              gradientOutput,
              gradientsWritten.get(gradientOutput),
            );
          }
          gradientOutputs.push(gradientOutput);
        }
      }

      if (intermediateVariables) {
        for (const inter of intermediateVariables) {
          if (!inputs.includes(inter)) {
            inputs.push(inter);
          }
        }
      }
      if (backpropCode.includes(intermediate(node))) {
        inputs.push(intermediate(node));
      }

      if (backpropCode !== "") {
        backwardCode += `${backpropCode}\n`;
      }
    }

    // Process dependencies (children) after processing the current node
    for (const dep of node.dependencies) {
      const output = node.variable.includes("intermediate")
        ? `grad_${node.variable}_output`
        : `grad_${node.variable}_intermediate_output`;
      generateBackwardCode(dep, `${output}[index]`);
    }
  };

  // Start with the root of the AST (final operation)
  console.log("backward generation finalNode=", finalNode.variable, finalNode.gradientVariable);
  generateBackwardCode(finalNode, finalNode.gradientVariable);
  inputs = Array.from(new Set(inputs));

  if (
    gradInit !== "1.0" &&
    (initializations.includes(trimIndex(gradInit)) || backwardCode.includes(trimIndex(gradInit)))
  ) {
    if (!initializations.includes(`${trimIndex(gradInit)}=`)) {
      const trimmedGradInit = trimIndex(gradInit);
      if (!inputs.includes(trimmedGradInit)) {
        inputs.push(trimmedGradInit);
      }
    }
  }

  for (const init of gradientInitializations) {
    if (backwardCode.includes(`let ${init}`)) continue;
    if (!saved.has(init)) initializations += `var ${init} = ${gradInit}; // saved inits\n`;
    if (inputs.includes(init)) {
      inputs = inputs.filter((x) => x !== init);
    }
    console.log("looking at init=", init, gradInit);
  }

  const inputsCode = inputs.map((input, index) => constructGroup(index, "read", input)).join("\n");
  console.log("inputs to construct", inputs);
  console.log(inputsCode);

  // Generate output code for leaf nodes (inputs)
  const visitedInputs = new Set<string>();
  for (const node of inputNodes) {
    if (!visitedInputs.has(node.variable)) {
      visitedInputs.add(node.variable);
      const output = `grad_${node.variable}_output`;
      if (!inputsCode.includes(output)) {
        //outputs.push(output);
        //outputCode += ` if (index < ${shapeToSize(getShape(node))}){ ${output}[index] = ${node.gradientVariable}; } \n`;
        //outputCode += ` ${output}[index] = ${node.gradientVariable}; \n`;
      }
    }
  }

  for (const gradientOutput of gradientOutputs) {
    const output = `${gradientOutput}_intermediate_output`;

    outputs.push(output);
    const l = Array.from(visited);
    const node = l.find((x) => output.startsWith(x.gradientVariable));
    outputCode += `
    // gradientsWritten.get() -> ${gradientsWrittenLocal.get(gradientOutput)}
    `;
    if (node) {
      // VERY IMPORTANT: ensure that we write w/in bounds, or else we might corrupt adjacent buffers!
      console.log(
        "size of node=",
        node.variable,
        output,
        gradientOutput,
        getShape(node),
        shapeToSize(getShape(node)),
        node,
      );
      if (
        gradientsWrittenLocal.has(gradientOutput) &&
        gradientsWrittenLocal.get(gradientOutput) > 1
      ) {
        outputCode += `
        if (index < ${shapeToSize(getShape(node))}) {
          let existing_value = ${output}[index]; // Read existing gradient
          ${output}[index] = existing_value + ${gradientOutput}; // Accumulate gradient
        }
      `;
      } else {
        outputCode += `if (index < ${shapeToSize(getShape(node))}){  ${output}[index] = ${gradientOutput}; } \n`;
      }
    } else {
      if (
        gradientsWrittenLocal.has(gradientOutput) &&
        gradientsWrittenLocal.get(gradientOutput) > 1
      ) {
        // Accumulation logic for unknown nodes
        outputCode += `
      let existing_value = ${output}[index]; // Read existing gradient
      ${output}[index] = existing_value + ${gradientOutput}; // Accumulate gradient
    `;
      } else {
        outputCode += `${output}[index] = ${gradientOutput}; \n`;
      }
    }
    console.log(
      "writing out gradientOutput=%s",
      gradientOutput,
      gradientsWrittenLocal.get(gradientOutput),
    );
  }

  for (const node of crossNodes) {
    if (!visitedInputs.has(node.variable)) {
      visitedInputs.add(node.variable);
      const output = node.variable;
      outputs.push(output);
      outputCode += `      ${output}[index] = ${node.gradientVariable};\n`;
    }
  }

  const outputsCode = Array.from(new Set(outputs))
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
