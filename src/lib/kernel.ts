import { Context } from "./context";
import { shapeToSize } from "./utils";
import { ASTNode } from "./zen";

export class Kernel {
  private pipeline: GPUComputePipeline;
  private bindGroup: GPUBindGroup;
  inputBuffers: Map<string, GPUBuffer>;
  outputBuffers: Map<string, GPUBuffer> = new Map();
  private intermediateBuffers: GPUBuffer[] = [];
  inputs: string[];
  outputs: string[];
  context?: Context<ASTNode>;
  kernelCode: string;
  size: number;

  constructor(
    private device: GPUDevice,
    kernelCode: string,
    inputs: string[],
    outputs: string[],
    intermediates: string[],
    inputBuffers: Map<string, GPUBuffer>,
    size: number,
    nodes: ASTNode[],
  ) {
    this.size = size;
    console.log("KERNEL size=", size, outputs, intermediates);
    this.inputs = inputs;
    this.outputs = outputs;
    this.inputBuffers = new Map(inputBuffers);
    this.kernelCode = kernelCode;

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

    console.log("nodes=", nodes, intermediates);
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
      console.log("setting input=%s for kernel", name, buffer, this);
      entries.push({
        binding: index,
        resource: { buffer: buffer! },
      });
    });

    // Create output buffers and add bindings
    outputs.forEach((name, index) => {
      const node = nodes.find(
        (x) =>
          `${x.gradientVariable}_intermediate_output` === name ||
          `${x.gradientVariable}_output` === name,
      );
      console.log("output nodes=%s", name, node, outputs, nodes, size);
      const _size = node?.shape ? shapeToSize(node.shape) : size;
      console.log("size determined for output=%s", name, _size);
      const buffer = device.createBuffer({
        size: _size * Float32Array.BYTES_PER_ELEMENT, // Assuming max size, adjust as needed
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
      const node = nodes.find((x) => x.variable === `${name}_intermediate`);
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
