// Define basic tensor types
type Tensor1D = number[];
type Tensor2D = number[][];

// Define supported operations
type Operation = "add" | "multiply" | "subtract" | "divide";

export class TinyTensorWebGPU {
	private device: GPUDevice;


	constructor(device: GPUDevice) {
		this.device = device;
	}

	// Generate shader code for a given operation
	private generateShader(op: Operation): string {
		const shaderCode = `
      @group(0) @binding(0) var<storage, read> input1: array<f32>;
      @group(0) @binding(1) var<storage, read> input2: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        let a = input1[index];
        let b = input2[index];
        output[index] = ${this.getOperationCode(op)};
      }
    `;
		return shaderCode;
	}

	// Helper method to get the operation code
	private getOperationCode(op: Operation): string {
		switch (op) {
			case "add":
				return "a + b";
			case "multiply":
				return "a * b";
			case "subtract":
				return "a - b";
			case "divide":
				return "a / b";
			default:
				throw new Error(`Unsupported operation: ${op}`);
		}
	}

	// Method to perform operation on 1D tensors
	async operate1D(a: Tensor1D, b: Tensor1D, op: Operation): Promise<Tensor1D> {
		if (a.length !== b.length) {
			throw new Error("Tensors must have the same length");
		}

		const shaderModule = this.device.createShaderModule({
			code: this.generateShader(op),
		});

		const bindGroupLayout = this.device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "storage" },
				},
			],
		});

		const pipelineLayout = this.device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		});

		const computePipeline = this.device.createComputePipeline({
			layout: pipelineLayout,
			compute: {
				module: shaderModule,
				entryPoint: "main",
			},
		});

		const input1Buffer = this.device.createBuffer({
			size: a.length * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		const input2Buffer = this.device.createBuffer({
			size: b.length * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		const outputBuffer = this.device.createBuffer({
			size: a.length * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.device.queue.writeBuffer(input1Buffer, 0, new Float32Array(a));
		this.device.queue.writeBuffer(input2Buffer, 0, new Float32Array(b));

		const bindGroup = this.device.createBindGroup({
			layout: bindGroupLayout,
			entries: [
				{ binding: 0, resource: { buffer: input1Buffer } },
				{ binding: 1, resource: { buffer: input2Buffer } },
				{ binding: 2, resource: { buffer: outputBuffer } },
			],
		});

		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.dispatchWorkgroups(Math.ceil(a.length / 64));
		passEncoder.end();

		const gpuCommands = commandEncoder.finish();
		this.device.queue.submit([gpuCommands]);

		// Create a separate command encoder for the copy operation
		const copyCommandEncoder = this.device.createCommandEncoder();
		const resultBuffer = this.device.createBuffer({
			size: a.length * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		copyCommandEncoder.copyBufferToBuffer(
			outputBuffer,
			0,
			resultBuffer,
			0,
			a.length * Float32Array.BYTES_PER_ELEMENT,
		);

		this.device.queue.submit([copyCommandEncoder.finish()]);

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const resultArray = new Float32Array(resultBuffer.getMappedRange());
		const result = Array.from(resultArray);
		resultBuffer.unmap();

		return result;
	}

	// Method to perform operation on 2D tensors
	async operate2D(a: Tensor2D, b: Tensor2D, op: Operation): Promise<Tensor2D> {
		// Flatten 2D tensors to 1D
		const flatA = a.flat();
		const flatB = b.flat();

		const flatResult = await this.operate1D(flatA, flatB, op);

		// Reshape the result back to 2D
		const result: Tensor2D = [];
		for (let i = 0; i < a.length; i++) {
			result.push(flatResult.slice(i * a[0].length, (i + 1) * a[0].length));
		}

		return result;
	}
}

// Example usage
async function main() {
	if (!navigator.gpu) {
		throw new Error("WebGPU not supported on this browser.");
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw new Error("No appropriate GPUAdapter found.");
	}

	const device = await adapter.requestDevice();
	const framework = new TinyTensorWebGPU(device);

	const a1D = [1, 2, 3, 4];
	const b1D = [5, 6, 7, 8];
	const result1D = await framework.operate1D(a1D, b1D, "add");
	console.log("1D Result:", result1D);

	const a2D = [
		[1, 2],
		[3, 4],
	];
	const b2D = [
		[5, 6],
		[7, 8],
	];
	const result2D = await framework.operate2D(a2D, b2D, "multiply");
	console.log("2D Result:", result2D);
}

//main().catch(console.error);
