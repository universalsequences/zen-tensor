import { KernelContext, Context } from "./context";
import { Kernel } from "./kernel";
import { OpType, Arg, Gen, ASTNode, DataType, toScalar } from "./zen";
import { Tensor } from "./input";
import { backpass } from "./back";

export class TensorGraph {
  device: GPUDevice;
  private contexts: Context<ASTNode>[] = [];
  kernels: Kernel[] = [];
  backKernels: Kernel[] = [];
  inputData: Map<string, Float32Array> = new Map();
  inputBuffers: Map<string, GPUBuffer> = new Map();
  private inputCounter: number = 0;
  outputSize: number = 0;
  outputShape: number[] = [1];
  backpasses: string[] = [];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  tensor(shape: number[]): Tensor {
    const tensorName = `tensor_${this.inputCounter++}`;
    const size = shape.reduce((a, b) => a * b, 1);
    const placeholder = new Tensor(tensorName, this, shape);
    this.inputData.set(tensorName, new Float32Array(size));
    return placeholder;
  }

  updateTensor(name: string, data: number[] | Float32Array) {
    const inputArray = data instanceof Float32Array ? data : new Float32Array(data);
    this.inputData.set(name, inputArray);

    if (this.inputBuffers.has(name)) {
      const buffer = this.inputBuffers.get(name)!;
      this.device.queue.writeBuffer(buffer, 0, inputArray);
    }
  }

  output(x: Arg): Gen {
    return (context: Context<ASTNode>) => {
      const result = context.gen(x);
      const [v] = context.useVariables("output");
      context.addOutput(v);
      const code = `${v}[index] = ${toScalar(result, DataType.Scalar)};`;
      return context.emit(v, code, OpType.Regular, this.outputShape, result);
    };
  }

  compile(graph: Gen, outputShape: number[]) {
    this.outputShape = outputShape;
    this.outputSize = outputShape.reduce((a, b) => a * b, 1);

    this.contexts = [];
    let currentContext: Context<ASTNode> = new KernelContext(OpType.Regular, this);
    // this.contexts.push(currentContext);

    const allContexts = new Set<Context<ASTNode>>();
    const visited = new Set<ASTNode>();
    const traverse = (node: ASTNode) => {
      allContexts.add(node.context);
      if (visited.has(node)) {
        return;
      }
      visited.add(node);
      if (node.context !== currentContext) {
        currentContext = node.context;
        if (everyDependencyMet(currentContext, this.contexts)) {
          this.contexts = [...this.contexts, currentContext];
        }
      }
      currentContext.code = [node.code, ...currentContext.code];
      node.dependencies.forEach(traverse);
    };

    const result = graph(currentContext);
    traverse(result);

    this.backpasses = backpass(result.dependencies[0]).map((x) => x.code);
    for (const bb of this.backpasses) console.log(bb);

    // brute force remaining contexts via dependencies...
    for (let i = 0; i < allContexts.size * 2; i++) {
      for (const c of allContexts) {
        if (!this.contexts.includes(c)) {
          if (everyDependencyMet(c, this.contexts)) {
            this.contexts.push(c);
          }
        }
      }
    }

    console.log("CONTEXTS=", this.contexts);

    console.log("input data = ", this.inputData);

    for (const c of this.contexts) {
      c.evalLazyInputs();
    }

    // Create input buffers
    this.inputData.forEach((data, name) => {
      const buffer = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buffer, 0, data);
      this.inputBuffers.set(name, buffer);
    });

    // Create kernels
    this.kernels = this.contexts.map((context) => {
      const code = context.generateKernel();
      context.kernelCode = code;
      const k = new Kernel(
        this.device,
        code,
        context.getInputs(),
        context.getOutputs(),
        context.intermediateOutputs,
        this.inputBuffers,
        this.outputSize,
      );
      k.context = context;
      return k;
    });

    for (const context of this.contexts) {
      const backward = context.backward;
      if (!backward) continue;
      for (const inp of backward.inputs) {
        if (!this.inputBuffers.has(inp)) {
          const data = new Float32Array(this.outputSize);
          this.inputData.set(inp, data);
          context.addInput(inp);
          console.log("adding input", inp);
          const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.device.queue.writeBuffer(buffer, 0, data);
          this.inputBuffers.set(inp, buffer);
        }
      }
    }

    // Create kernels
    for (const context of this.contexts) {
      const backward = context.backward;
      if (!backward) continue;

      console.log("backward inputs", backward.inputs);
      console.log("backward outputs", backward.outputs);
      console.log(this.inputData);
      console.log(this.inputBuffers);
      const k = new Kernel(
        this.device,
        backward.code,
        backward.inputs,
        backward.outputs,
        [],
        this.inputBuffers,
        this.outputSize,
      );
      this.backKernels.push(k);
    }
  }

  async run() {
    const WORKGROUP_SIZE = 64; // This should match your shader's workgroup size
    const commandEncoder = this.device.createCommandEncoder();

    const kernels = [...this.kernels, ...this.backKernels];
    for (let i = 0; i < kernels.length; i++) {
      const commandEncoder = this.device.createCommandEncoder();
      const kernel = kernels[i];
      // If this is not the first kernel, we need to copy data from the previous kernel
      if (i > 0) {
        for (let j = 0; j < i; j++) {
          const prevOutputs = kernels[j].getOutputBuffers();
          const currentInputs = kernel.inputs;
          for (const inputName of currentInputs) {
            console.log("inputName=%s", inputName, prevOutputs);
            const slicedInputName = inputName.slice(0, inputName.length - "_intermediate".length);
            console.log("sliced input name=", slicedInputName, prevOutputs);
            if (inputName.includes("intermediate") && prevOutputs.has(slicedInputName)) {
              const sourceBuffer = prevOutputs.get(slicedInputName)!;
              const destBuffer = kernel.getInputBuffer(inputName)!;
              commandEncoder.copyBufferToBuffer(
                sourceBuffer,
                0,
                destBuffer,
                0,
                this.outputSize * Float32Array.BYTES_PER_ELEMENT,
              );
            }
            if (prevOutputs.has(inputName + "_out")) {
              const sourceBuffer = prevOutputs.get(inputName + "_out")!;
              const destBuffer = kernel.getInputBuffer(inputName)!;
              // await logBuffer(this.device, sourceBuffer, `Kernel ${i} input ${inputName}`);

              commandEncoder.copyBufferToBuffer(
                sourceBuffer,
                0,
                destBuffer,
                0,
                this.outputSize * Float32Array.BYTES_PER_ELEMENT,
              );
            }
          }
        }
      }

      // Calculate the number of workgroups needed
      const numWorkgroups = Math.ceil(this.outputSize / WORKGROUP_SIZE);

      // Run the current kernel with the calculated number of workgroups
      kernel.run(commandEncoder, numWorkgroups);

      this.device.queue.submit([commandEncoder.finish()]);
    }

    const grads = new Map<string, Float32Array>();
    for (const kernel of kernels) {
      for (const output of kernel.outputs) {
        if (!output.includes("grad")) {
          continue;
        }
        const commandEncoder = this.device.createCommandEncoder();
        const destBuffer = kernel.getOutputBuffer(output);
        if (destBuffer) {
          console.log("trying to fetch buffer for output=", output);
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
          console.log("RESULT ARRAY FOR KERNEL", resultArray);
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

async function logBuffer(device: GPUDevice, buffer: GPUBuffer, label: string) {
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
  console.log(`${label}:`, Array.from(data));
  stagingBuffer.unmap();
}

const everyDependencyMet = (context: Context<ASTNode>, contexts: Context<ASTNode>[]) => {
  for (const input of context.inputs.keys()) {
    if (input.includes("tensor")) {
      continue;
    }
    if (
      !contexts.some((c) => {
        return c.outputs.has(input + "_out");
      })
    ) {
      return false;
    }
  }
  return true;
};
