import { KernelContext } from "./context";

export class Kernel {
	context: KernelContext;
	private pipeline: GPUComputePipeline;
	private bindGroup: GPUBindGroup;
	private inputBuffers: Map<string, GPUBuffer>;
	private outputBuffers: Map<string, GPUBuffer> = new Map();

	constructor(
		private device: GPUDevice,
		context: KernelContext,
		inputBuffers: Map<string, GPUBuffer>,
	) {
		this.context = context;
		this.inputBuffers = new Map(inputBuffers);

		const code = context.getShaderCode();
		console.log("KERNEL_BEGIN:");
		console.log(code);
		console.log("KERNEL_END");
		const shaderModule = device.createShaderModule({
			code,
		});
		console.log("SUCCESS");

		this.pipeline = device.createComputePipeline({
			layout: "auto",
			compute: {
				module: shaderModule,
				entryPoint: "main",
			},
		});

		const entries: GPUBindGroupEntry[] = [];

		// Add input bindings
		context.getInputs().forEach((name, index) => {
			entries.push({
				binding: index,
				resource: { buffer: this.inputBuffers.get(name)! },
			});
		});

		// Create output buffers and add bindings
		const outputs = context.getOutputs();
		console.log("kernel constructor outputs=", this, outputs);
		outputs.forEach((name, index) => {
			const buffer = device.createBuffer({
				size: 1024 * Float32Array.BYTES_PER_ELEMENT, // Assuming max size, adjust as needed
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			});
			this.outputBuffers.set(name, buffer);
			console.log("create buffer = ", buffer);
			entries.push({
				binding: context.getInputs().length + index,
				resource: { buffer },
			});
		});
		console.log("entries=", entries);

		this.bindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries,
		});
		console.log("finished bind");
	}

	getInputBuffer(name: string): GPUBuffer | undefined {
		return this.inputBuffers.get(name);
	}

	run(commandEncoder: GPUCommandEncoder, size: number) {
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);
		passEncoder.dispatchWorkgroups(Math.ceil(size / 64));
		passEncoder.end();
	}

	getOutputBuffer(name?: string): GPUBuffer | undefined {
		if (!name) {
			console.log("getting final buffer", name, this.outputBuffers.keys());
			for (const key of this.outputBuffers.keys()) {
				return this.outputBuffers.get(key);
			}
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
		this.context.getInputs().forEach((inputName, index) => {
			entries.push({
				binding: index,
				resource: { buffer: this.inputBuffers.get(inputName)! },
			});
		});
		this.context.getOutputs().forEach((outputName, index) => {
			entries.push({
				binding: this.context.getInputs().length + index,
				resource: { buffer: this.outputBuffers.get(outputName)! },
			});
		});
		this.bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries,
		});
	}
}
