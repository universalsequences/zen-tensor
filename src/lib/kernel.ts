import { Context } from "./context";
import { ASTNode } from "./zen";

export class Kernel {
  private pipeline: GPUComputePipeline;
  private bindGroup: GPUBindGroup;
  private inputBuffers: Map<string, GPUBuffer>;
  private outputBuffers: Map<string, GPUBuffer> = new Map();
  private intermediateBuffers: GPUBuffer[] = [];
  private inputs: string[];
  private outputs: string[];
  context?: Context<ASTNode>;

  constructor(
    private device: GPUDevice,
    kernelCode: string,
    inputs: string[],
    outputs: string[],
    intermediates: string[],
    inputBuffers: Map<string, GPUBuffer>,
    size: number,
  ) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.inputBuffers = new Map(inputBuffers);
    const shaderModule = device.createShaderModule({
      code: kernelCode,
    });

    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    const entries: GPUBindGroupEntry[] = [];

    console.log("KERNEL inputBuffers=", inputBuffers);
    // Add input bindings
    inputs.forEach((name, index) => {
      const buffer = this.inputBuffers.get(name);
      console.log("buffer for name=%s", name, buffer);
      entries.push({
        binding: index,
        resource: { buffer: buffer! },
      });
    });

    // Create output buffers and add bindings
    outputs.forEach((name, index) => {
      const buffer = device.createBuffer({
        size: size * Float32Array.BYTES_PER_ELEMENT, // Assuming max size, adjust as needed
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.outputBuffers.set(name, buffer);
      entries.push({
        binding: inputs.length + index,
        resource: { buffer },
      });
    });

    intermediates.forEach((name, index) => {
      const buffer = device.createBuffer({
        size: size * Float32Array.BYTES_PER_ELEMENT, // Assuming max size, adjust as needed
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.outputBuffers.set(name, buffer);
      this.intermediateBuffers.push(buffer);
      entries.push({
        binding: inputs.length + outputs.length + index,
        resource: { buffer },
      });
    });

    console.log("bindgroup entries", entries);
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    });
  }

  getInputBuffer(name: string): GPUBuffer | undefined {
    return this.inputBuffers.get(name);
  }

  run(commandEncoder: GPUCommandEncoder, numWorkgroups: number) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
  }

  getOutputBuffer(name?: string): GPUBuffer | undefined {
    if (!name) {
      for (const key of this.outputBuffers.keys()) {
        return this.outputBuffers.get(key);
      }
      return undefined;
    }
    return this.outputBuffers.get(name);
  }

  getOutputBuffers(): Map<string, GPUBuffer> {
    return this.outputBuffers;
  }

  updateInputBuffer(name: string, buffer: GPUBuffer) {
    this.inputBuffers.set(name, buffer);
    // Recreate bind group with updated input buffer
    const entries: GPUBindGroupEntry[] = [];
    this.inputs.forEach((inputName, index) => {
      entries.push({
        binding: index,
        resource: { buffer: this.inputBuffers.get(inputName)! },
      });
    });
    this.outputs.forEach((outputName, index) => {
      entries.push({
        binding: this.inputs.length + index,
        resource: { buffer: this.outputBuffers.get(outputName)! },
      });
    });
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    });
  }
}
