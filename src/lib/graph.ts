import { KernelContext, Context } from "./context";
import { Kernel } from "./kernel";
import { OpType, Arg, Gen, ASTNode, DataType, toScalar } from "./zen";
import { Tensor } from "./input";

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

	tensor(shape: number[]): Tensor {
		const tensorName = `tensor_${this.inputCounter++}`;
		const size = shape.reduce((a, b) => a * b, 1);
		const placeholder = new Tensor(tensorName, this, shape);
		this.inputData.set(tensorName, new Float32Array(size));
		return placeholder;
	}

	updateTensor(name: string, data: number[] | Float32Array) {
		console.log("updating tensor name=%s", name, data);
		const inputArray =
			data instanceof Float32Array ? data : new Float32Array(data);
		this.inputData.set(name, inputArray);

		if (this.inputBuffers.has(name)) {
			const buffer = this.inputBuffers.get(name)!;
			console.log("writing to buffer", name, buffer, inputArray);
			this.device.queue.writeBuffer(buffer, 0, inputArray);
		} else {
			console.log("could not write to buffer", name);
		}
	}

	output(x: Arg): Gen {
		return (context: Context) => {
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
		let currentContext = new KernelContext(OpType.Regular, this);
		this.contexts.push(currentContext);

		const traverse = (node: ASTNode) => {
			if (node.context !== currentContext) {
				currentContext = node.context;
				this.contexts = [currentContext, ...this.contexts];
			}
			currentContext.code = [node.code, ...currentContext.code];
			node.dependencies.forEach(traverse);
		};

		const result = graph(currentContext);
		console.log("graph result=", result);
		traverse(result);

		// Create input buffers
		this.inputData.forEach((data, name) => {
			const buffer = this.device.createBuffer({
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.device.queue.writeBuffer(buffer, 0, data);
			console.log("input data=", data, name, buffer);
			this.inputBuffers.set(name, buffer);
		});

		// Create kernels
		this.kernels = this.contexts.map(
			(context) =>
				new Kernel(this.device, context, this.inputBuffers, this.outputSize),
		);
	}

	async run(): Promise<Float32Array> {
		console.log("Kernels we are running=", [...this.kernels]);
		const commandEncoder = this.device.createCommandEncoder();
		const WORKGROUP_SIZE = 64; // This should match your shader's workgroup size

		for (let i = 0; i < this.kernels.length; i++) {
			const kernel = this.kernels[i];
			console.log("kernels i=0 *******", i);
			// If this is not the first kernel, we need to copy data from the previous kernel
			if (i > 0) {
				for (let j = 0; j < i; j++) {
					console.log("looking at kernels j=0", j);
					const prevOutputs = this.kernels[j].getOutputBuffers();
					const currentInputs = kernel.context.getInputs();
					for (const inputName of currentInputs) {
						if (prevOutputs.has(inputName + "_out")) {
							const sourceBuffer = prevOutputs.get(inputName + "_out")!;
							const destBuffer = kernel.getInputBuffer(inputName)!;
							console.log("source buffer=", sourceBuffer);
							console.log("dest buffer=", destBuffer);
							await logBuffer(
								this.device,
								sourceBuffer,
								`Kernel ${i} input ${inputName}`,
							);

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

		// Get the final outpt
		const finalKernel = this.kernels[this.kernels.length - 1];
		const finalOutputBuffer = finalKernel.getOutputBuffer();
		console.log("kernels = ", this.kernels, finalOutputBuffer);
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

		return resultArray;
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
