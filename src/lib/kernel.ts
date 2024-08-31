import { Context } from "./context";
import { ASTNode } from "./zen";

export class Kernel {
  private pipeline: GPUComputePipeline;
  private bindGroup: GPUBindGroup;
  private inputBuffers: Map<string, GPUBuffer>;
  private outputBuffers: Map<string, GPUBuffer> = new Map();
  private intermediateBuffers: GPUBuffer[] = [];
  inputs: string[];
  outputs: string[];
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

    // Create the shader module
    const shaderModule = device.createShaderModule({
      code: kernelCode,
    });

    // Create the bind group layout explicitly
    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];

    // Add input buffer layout entries
    inputs.forEach((name, index) => {
      bindGroupLayoutEntries.push({
        binding: index,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      });
    });

    // Add output buffer layout entries
    outputs.forEach((name, index) => {
      bindGroupLayoutEntries.push({
        binding: inputs.length + index,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      });
    });

    // Add intermediate buffer layout entries
    intermediates.forEach((name, index) => {
      bindGroupLayoutEntries.push({
        binding: inputs.length + outputs.length + index,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      });
    });

    // Create the bind group layout with all entries
    const bindGroupLayout = device.createBindGroupLayout({
      entries: bindGroupLayoutEntries,
    });

    // Create the compute pipeline with the explicitly created layout
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    // Create bind group entries
    const entries: GPUBindGroupEntry[] = [];

    // Add input bindings
    inputs.forEach((name, index) => {
      const buffer = this.inputBuffers.get(name);
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

    // Add intermediate buffer bindings
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

    // Create the bind group with the explicitly created layout
    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
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
