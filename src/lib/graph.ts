import { KernelContext, Context } from "./context";
import { Kernel } from "./kernel";
import { OpType, Arg, Gen, GenResult, Type, variable } from "./zen";
import { InputPlaceholder } from "./input";
//

export class TensorGraph {
	device: GPUDevice;
	private contexts: KernelContext[] = [];
	kernels: Kernel[] = [];
	private inputData: Map<string, Float32Array> = new Map();
	private inputBuffers: Map<string, GPUBuffer> = new Map();
	private inputCounter: number = 0;
	outputSize: number = 0;
	outputShape: number[] = [1];

	constructor(device: GPUDevice) {
		this.device = device;
	}

	input(shape: number[]): InputPlaceholder {
		const inputName = `input_${this.inputCounter++}`;
		const size = shape.reduce((a, b) => a * b, 1);
		const placeholder = new InputPlaceholder(inputName, this, shape);
		this.inputData.set(inputName, new Float32Array(size));
		return placeholder;
	}

	updateInput(name: string, data: number[] | Float32Array) {
		const inputArray =
			data instanceof Float32Array ? data : new Float32Array(data);
		this.inputData.set(name, inputArray);

		if (this.inputBuffers.has(name)) {
			const buffer = this.inputBuffers.get(name)!;
			this.device.queue.writeBuffer(buffer, 0, inputArray);
		}
	}

	output(x: Arg): Gen {
		return (context: Context) => {
			const result = context.gen(x);
			const [v] = context.useVariables("output");
			context.addOutput(v);
			const code = `${v}[index] = ${variable(result, Type.Scalar)};`;
			return context.emit(v, code, OpType.Regular, this.outputShape, result);
		};
	}

	compile(graph: Gen, outputShape: number[]) {
		this.outputShape = outputShape;
		this.outputSize = outputShape.reduce((a, b) => a * b, 1);

		this.contexts = [];
		let currentContext = new KernelContext(
			OpType.Regular,
			this,
		);
		this.contexts.push(currentContext);

		const traverse = (node: GenResult) => {
			if (node.context !== currentContext) {
				currentContext = node.context;
				this.contexts = [currentContext, ...this.contexts];
			}
			currentContext.code = [node.code, ...currentContext.code];
			node.dependencies.forEach(traverse);
		};

		const result = graph(currentContext);
		traverse(result);

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
		this.kernels = this.contexts.map(
			(context) => new Kernel(this.device, context, this.inputBuffers, this.outputSize),

		);
	}

	private async readBuffer(
		buffer: GPUBuffer,
		size: number,
	): Promise<Float32Array> {
		const readBuffer = this.device.createBuffer({
			size: size * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyBufferToBuffer(
			buffer,
			0,
			readBuffer,
			0,
			size * Float32Array.BYTES_PER_ELEMENT,
		);

		this.device.queue.submit([commandEncoder.finish()]);

		await readBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = readBuffer.getMappedRange();
		const resultArray = new Float32Array(arrayBuffer.slice(0));
		readBuffer.unmap();
		readBuffer.destroy();

		return resultArray;
	}

	async run2(): Promise<Float32Array> {
		const commandEncoder = this.device.createCommandEncoder();

		for (let i = 0; i < this.kernels.length; i++) {
			const kernel = this.kernels[i];

			// If this is not the first kernel, we need to copy data from the previous kernel
			if (i > 0) {
				const prevKernel = this.kernels[i - 1];
				for (let j = 0; j < i; j++) {
					const prevOutputs = this.kernels[j].getOutputBuffers();
					//const prevOutputs = prevKernel.getOutputBuffers();
					const currentInputs = kernel.context.getInputs();

					for (const inputName of currentInputs) {
						if (prevOutputs.has(inputName + "_out")) {
							const sourceBuffer = prevOutputs.get(inputName + "_out")!;
							const destBuffer = kernel.getInputBuffer(inputName)!;

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

			// Run the current kernel
			kernel.run(commandEncoder, this.outputSize);
		}

		// Get the final output
		const finalKernel = this.kernels[this.kernels.length - 1];
		const finalOutputBuffer = finalKernel.getOutputBuffer();

		if (!finalOutputBuffer) {
			throw new Error("Final output buffer not found");
		}

		// Copy final output to a readable buffer
		const resultBuffer = this.device.createBuffer({
			size: this.outputSize * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		commandEncoder.copyBufferToBuffer(
			finalOutputBuffer,
			0,
			resultBuffer,
			0,
			this.outputSize * Float32Array.BYTES_PER_ELEMENT,
		);

		// Submit all command encoder operations
		this.device.queue.submit([commandEncoder.finish()]);

		// Read the result
		await resultBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = resultBuffer.getMappedRange();
		const resultArray = new Float32Array(arrayBuffer.slice(0));
		resultBuffer.unmap();

		resultBuffer.destroy();

		return resultArray;
	}

	async run(): Promise<Float32Array> {
		const commandEncoder = this.device.createCommandEncoder();
		const WORKGROUP_SIZE = 64; // This should match your shader's workgroup size

		for (let i = 0; i < this.kernels.length; i++) {
			const kernel = this.kernels[i];
			// If this is not the first kernel, we need to copy data from the previous kernel
			if (i > 0) {
				for (let j = 0; j < i; j++) {
					const prevOutputs = this.kernels[j].getOutputBuffers();
					const currentInputs = kernel.context.getInputs();
					for (const inputName of currentInputs) {
						if (prevOutputs.has(inputName + "_out")) {
							const sourceBuffer = prevOutputs.get(inputName + "_out")!;
							const destBuffer = kernel.getInputBuffer(inputName)!;
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
		}

		// Get the final output
		const finalKernel = this.kernels[this.kernels.length - 1];
		const finalOutputBuffer = finalKernel.getOutputBuffer();
		if (!finalOutputBuffer) {
			throw new Error("Final output buffer not found");
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

		return resultArray;
	}

	// Helper method to determine the expected result length
	private getExpectedResultLength(): number {
		const finalKernel = this.kernels[this.kernels.length - 1];
		return this.outputSize;
	}

	destroy() {
		this.inputBuffers.forEach((buffer) => buffer.destroy());
		this.kernels.forEach((kernel) => {
			kernel.getOutputBuffer("output").destroy();
		});
	}
}
