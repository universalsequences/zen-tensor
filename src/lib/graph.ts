import { KernelContext, Context } from "./context";
import { Kernel } from "./kernel";
import { OpType, Arg, Gen, ASTNode, DataType, toScalar } from "./zen";
import { Tensor } from "./tensor";
import { backpass } from "./back";
import { shapeToSize } from "./utils";
import { getShape } from "./reshape";

export interface GraphResult {
  forward: Float32Array;
  gradients: Map<string, Float32Array>;
}

export class TensorGraph {
  device: GPUDevice;
  private contexts: Context<ASTNode>[] = [];
  kernels: Kernel[] = [];
  backKernels: Kernel[] = [];
  inputData: Map<string, Float32Array> = new Map();
  gradientData: Map<string, Float32Array> = new Map();
  inputBuffers: Map<string, GPUBuffer> = new Map();
  private inputCounter: number = 0;
  outputSize: number = 0;
  outputShape: number[] = [1];
  backpasses: string[] = [];
  tensors: Map<string, Tensor> = new Map();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  tensor(shape: number[], name = ""): Tensor {
    const tensorName = `tensor_${name}_${this.inputCounter++}`;
    const size = shape.reduce((a, b) => a * b, 1);
    const placeholder = new Tensor(tensorName, this, shape);
    this.inputData.set(tensorName, new Float32Array(size));
    this.tensors.set(name, placeholder);
    return placeholder;
  }

  updateTensor(name: string, data: number[] | Float32Array) {
    const inputArray = data instanceof Float32Array ? data : new Float32Array(data);
    this.inputData.set(name, inputArray);

    if (this.inputBuffers.has(name)) {
      const buffer = this.inputBuffers.get(name)!;
      this.device.queue.writeBuffer(buffer, 0, inputArray);
    } else {
    }
  }

  output(x: Arg): Gen {
    return (context: Context<ASTNode>) => {
      const result = context.gen(x);
      const [v] = context.useVariables("output");
      context.addOutput(v);
      const code = `${v}[index] = ${toScalar(result)};`;
      return context.emit("output", v, code, OpType.Regular, this.outputShape, result);
    };
  }

  /**
   * Compiles an expression into series of kernel
   * */
  compile(graph: Gen, outputShape: number[]) {
    this.outputShape = outputShape;
    this.outputSize = outputShape.reduce((a, b) => a * b, 1);

    this.contexts = [];
    let currentContext: Context<ASTNode> = new KernelContext(OpType.Regular, this);
    // this.contexts.push(currentContext);

    const allContexts = new Set<Context<ASTNode>>();
    const visited = new Set<ASTNode>();

    // recursively traverse the AST to determine all the contexts and concatenate the kernel
    // code for each context
    const traverse = (node: ASTNode) => {
      allContexts.add(node.context);
      if (visited.has(node)) {
        return;
      }
      visited.add(node);
      if (node.context !== currentContext) {
        currentContext = node.context;
      }
      currentContext.code = [node.code, ...currentContext.code];
      node.dependencies.forEach(traverse);
    };

    const rootNode: ASTNode = graph(currentContext);

    traverse(rootNode);

    this.backpasses = backpass(rootNode.dependencies[0]).map((x) => x.code);

    for (const c of allContexts) {
      const code = c.generateKernel();
      c.kernelCode = code;
    }
    // brute force remaining contexts via dependencies...
    for (let i = 0; i < allContexts.size * 24; i++) {
      for (const c of allContexts) {
        if (!this.contexts.includes(c)) {
          if (everyDependencyMet(c, this.contexts, true)) {
            this.contexts.push(c);
          }
        }
      }
    }

    for (const c of this.contexts) {
      c.evalLazyInputs();
    }

    // Create input buffers for each context
    // NOTE - all contexts call back to the graph to create lazy "buffer" instances w/
    // the data to be used upon intialization
    this.inputData.forEach((data, name) => {
      const buffer = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buffer, 0, data);
      this.inputBuffers.set(name, buffer);
    });

    // Create forward-pass kernels from each context, generating the final kernel code as well as
    // setting up the binding ports for each kernel
    this.kernels = this.contexts.map((context, index) => {
      const code = context.generateKernel();
      context.kernelCode = code;
      const kernel = new Kernel(
        this.device,
        code,
        context.getInputs(),
        context.getOutputs(),
        context.intermediateOutputs,
        this.inputBuffers,
        context.size,
        context.nodes,
      );
      kernel.context = context;
      return kernel;
    });

    // setup the data for each backward-pass kernel
    for (const context of this.contexts) {
      const backward = context.backward;
      if (!backward) continue;
      for (const input of backward.inputs) {
        if (!this.inputBuffers.has(input)) {
          // create empty buffers for each backward input -- these are likely intermediate
          // values that will later be generated by the forward pass
          const node = context.nodes.find((x) => `grad_${x.parent?.variable}_output` === input);
          const size = node?.parent ? shapeToSize(getShape(node?.parent)) : context.size;
          const data = new Float32Array(size);
          this.inputData.set(input, data);
          context.addInput(input);
          const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.device.queue.writeBuffer(buffer, 0, data);
          this.inputBuffers.set(input, buffer);
        }
      }
    }

    // Create backward-pass kernels from each context (w/ a generated backward pass)
    let q = 0;
    for (const context of this.contexts) {
      const backward = context.backward;
      if (!backward) continue;

      q++;
      const kernel = new Kernel(
        this.device,
        backward.code,
        backward.inputs,
        backward.outputs,
        [],
        this.inputBuffers,
        context.size,
        context.nodes,
      );
      this.backKernels = [kernel, ...this.backKernels];
    }

    return {
      forward: this.kernels,
      backwards: this.backKernels,
    };
  }

  /**
   * Executes the forward and backwards kernels, passing data to and from these kernels
   * @returns the result of the computation
   * */
  async run(): Promise<GraphResult> {
    const WORKGROUP_SIZE = 64; // should match shader's workgroup size
    const commandEncoder = this.device.createCommandEncoder();

    // execute each kernel, passing data required by each kernel, from other kernels
    // (if they are generated by other kernels)
    const kernels = [...this.kernels, ...this.backKernels];
    for (let i = 0; i < kernels.length; i++) {
      const commandEncoder = this.device.createCommandEncoder();
      const currentKernel = kernels[i];

      // If this is not the first kernel, we need to copy data from the previous kernel
      if (i > 0) {
        for (let j = 0; j < i; j++) {
          // search previously run kernels for outputs that match an input from this kernel
          const previousKernel = kernels[j];
          const prevOutputs = previousKernel.getOutputBuffers();
          const currentInputs = currentKernel.inputBuffers.keys();
          for (const inputName of currentInputs) {
            const intermediateInputName = inputName.slice(
              0,
              inputName.length - "_intermediate".length,
            );
            if (inputName.includes("intermediate") && prevOutputs.has(intermediateInputName)) {
              // intermediate case
              const sourceBuffer = prevOutputs.get(intermediateInputName)!;
              const destBuffer = currentKernel.getInputBuffer(inputName)!;
              let len = this.inputData.get(inputName)?.length || this.outputSize;
              commandEncoder.copyBufferToBuffer(
                sourceBuffer,
                0,
                destBuffer,
                0,
                len * Float32Array.BYTES_PER_ELEMENT,
              );
              const r = async () => await logBuffer(this.device, sourceBuffer); //.then((r) => {
              if (previousKernel.context) {
                const astNodes = previousKernel.context?.nodes; //.filter((n) => n.variable === inputName) || [];
                for (const node of astNodes) {
                  node.result = r;
                }
              }
            }

            // gradient-output case (copying output of gradient kernel to other gradient kernel)
            if (prevOutputs.has(inputName)) {
              // we have a direct match between inputName and previous kernel output
              const sourceBuffer = prevOutputs.get(inputName)!;
              const destBuffer = currentKernel.getInputBuffer(inputName)!;

              commandEncoder.copyBufferToBuffer(
                sourceBuffer,
                0,
                destBuffer,
                0,
                Math.min(sourceBuffer.size, destBuffer.size),
              );
              // const r = await logBuffer(this.device, sourceBuffer);
            }
          }
        }
      }

      // Calculate the number of workgroups needed
      const size = currentKernel.size;
      const numWorkgroups = Math.ceil(size / WORKGROUP_SIZE);

      // Run the current kernel with the calculated number of workgroups
      // Note: this adds the execution of the current run to the "commandEncoder"
      currentKernel.run(commandEncoder, numWorkgroups);

      // this executes the commands encoded in the commandEncoder: data copies + kernel execution
      this.device.queue.submit([commandEncoder.finish()]);
    }

    // run each backpropagation kernel and save the gradients of the outputs,
    // which will be used to "learn" the weights (by comparing loss of entire system and multiplying
    // by gradient -- see Tensor.learn)
    const grads = new Map<string, Float32Array>();
    for (const kernel of kernels) {
      for (const output of kernel.outputs) {
        if (!output.includes("grad")) {
          continue;
        }

        const commandEncoder = this.device.createCommandEncoder();
        const destBuffer = kernel.getOutputBuffer(output);
        if (destBuffer) {
          const resultBuffer = this.device.createBuffer({
            size: destBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          commandEncoder.copyBufferToBuffer(destBuffer, 0, resultBuffer, 0, destBuffer.size);

          // Submit all command encoder operations
          this.device.queue.submit([commandEncoder.finish()]);

          // Read the result
          await resultBuffer.mapAsync(GPUMapMode.READ);
          const arrayBuffer = resultBuffer.getMappedRange();
          const resultArray = new Float32Array(arrayBuffer.slice(0));
          resultBuffer.unmap();
          resultBuffer.destroy();
          grads.set(output, resultArray);
        }
      }
    }

    // Get the final outpt
    const finalKernel = this.kernels[this.kernels.length - 1];
    const finalOutputBuffer = finalKernel.getOutputBuffer();
    if (!finalOutputBuffer) {
      throw new Error("Final output buffer not foundxyz FUCKKKKK");
    }

    // Copy final output to a readable buffer
    const resultBuffer = this.device.createBuffer({
      size: finalOutputBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    commandEncoder.copyBufferToBuffer(
      finalOutputBuffer,
      0,
      resultBuffer,
      0,
      finalOutputBuffer.size,
    );

    // Submit all command encoder operations
    this.device.queue.submit([commandEncoder.finish()]);

    // Read the result
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = resultBuffer.getMappedRange();
    const resultArray = new Float32Array(arrayBuffer.slice(0));
    resultBuffer.unmap();
    resultBuffer.destroy();

    for (const gradInput of grads.keys()) {
      const regex = /^grad_(.+?)_output$/;
      const match = gradInput.match(regex);
      if (match) {
        const extractedPart = match[1];
        const gradient = grads.get(gradInput)!;
        this.gradientData.set(extractedPart, gradient);
        for (const kernel of kernels) {
          const { context } = kernel;
          if (context && gradient) {
            const nodes = context.nodes.filter((node) => node.variable === extractedPart);
            for (const node of nodes) {
              node.gradient = Array.from(gradient);
            }
          }
        }
      }
    }

    return {
      forward: resultArray,
      gradients: grads,
    };
  }

  destroy() {
    this.inputBuffers.forEach((buffer) => buffer.destroy());
    this.kernels.forEach((kernel) => {
      kernel.getOutputBuffer("output")?.destroy();
    });
  }
}

async function logBuffer(device: GPUDevice, buffer: GPUBuffer) {
  const stagingBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const data = new Float32Array(copyArrayBuffer);
  const x = Array.from(data);
  stagingBuffer.unmap();
  return x;
}

const everyDependencyMet = (
  context: Context<ASTNode>,
  previousContexts: Context<ASTNode>[],
  lazy = false,
) => {
  for (const input of context.inputs.keys()) {
    if (input.includes("tensor")) {
      continue;
    }
    if (
      !previousContexts.some((c) => {
        return c.outputs.has(input + "_out");
      })
    ) {
      return false;
    }
  }
  if (!lazy) {
    return true;
  }
  for (const input of context.lazyInputs) {
    if (
      !previousContexts.some((c) => {
        return c.intermediateOutputs.includes(input);
      })
    ) {
      return false;
    }
  }
  return true;
};
