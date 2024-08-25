enum OpType {
	Regular,
	Reduction,
}

type Gen = (context: Context) => GenResult;

interface GenResult {
	variable: string;
	code: string;
	dependencies: GenResult[];
	opType: OpType;
}

interface Context {
	gen: (x: Gen) => GenResult;
	useVariables: (...names: string[]) => string[];
	emit: (
		variable: string,
		code: string,
		opType: OpType,
		...dependencies: GenResult[]
	) => GenResult;
	useContext: (opType: OpType) => Context;
	getBindingIndex: (name: string) => number;
}

class ShaderBuilder implements Context {
	private code: string[] = [];
	private inputs: Map<string, number> = new Map();
	private output: string | null = null;

	private currentOpType: OpType = OpType.Regular;
	private idx = 0;

	gen(x: Gen): GenResult {
		return x(this);
	}

	useVariables(...names: string[]) {
		this.idx++;
		return names.map((n) => `${n}${this.idx}`);
	}

	emit(
		variable: string,
		code: string,
		opType: OpType,
		...dependencies: GenResult[]
	): GenResult {
		this.code.push(code);
		return { variable, code, dependencies, opType };
	}

	useContext(opType: OpType): Context {
		if (this.currentOpType !== opType) {
			this.currentOpType = opType;
			this.code.push(`// Switching to ${OpType[opType]} context`);
		}
		return this;
	}

	addInput(name: string) {
		if (!this.inputs.has(name)) {
			this.inputs.set(name, this.inputs.size);
		}
	}

	setOutput(name: string) {
		this.output = name;
	}

	getBindingIndex(name: string): number {
		if (this.inputs.has(name)) {
			return this.inputs.get(name)!;
		}
		if (name === this.output) {
			return this.inputs.size;
		}
		throw new Error(`Binding index not found for ${name}`);
	}

	getShaderCode(): string {
		const inputBindings = Array.from(this.inputs.entries())
			.map(
				([name, index]) =>
					`@group(0) @binding(${index}) var<storage, read> ${name}: array<f32>;`,
			)
			.join("\n");

		const outputBinding = this.output
			? `@group(0) @binding(${this.inputs.size}) var<storage, read_write> ${this.output}: array<f32>;`
			: "";

		return `
      ${inputBindings}
      ${outputBinding}

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        ${this.code.join("\n")}
      }
    `;
	}
}

// Curried function for binary operations
const binaryOp =
	(name: string, op: string) =>
	(x: Gen, y: Gen) =>
	(context: Context): GenResult => {
		const _x = context.gen(x);
		const _y = context.gen(y);
		const [variableName] = context.useVariables(`${name}_result`);
		const code = `let ${variableName} = ${_x.variable} ${op} ${_y.variable};`;
		return context.emit(variableName, code, OpType.Regular, _x, _y);
	};

// Define operations
export const add = binaryOp("add", "+");
export const mult = binaryOp("mult", "*");
export const sub = binaryOp("sub", "-");
export const div = binaryOp("div", "/");

// Reduction operation
export const reduce =
	(op: string) =>
	(x: Gen) =>
	(context: Context): GenResult => {
		const reductionContext = context.useContext(OpType.Reduction);
		const _x = reductionContext.gen(x);
		const [variableName] = reductionContext.useVariables(`${op}_result`);
		const code = `
    var ${variableName} = ${_x.variable}[0];
    for (var i = 1u; i < arrayLength(&${_x.variable}); i = i + 1u) {
      ${variableName} = ${variableName} ${op} ${_x.variable}[i];
    }
  `;
		return reductionContext.emit(variableName, code, OpType.Reduction, _x);
	};

// Audio-specific operations
export const sine =
	(freq: Gen) =>
	(context: Context): GenResult => {
		const [variableName] = context.useVariables("sine_wave");
		const _freq = context.gen(freq);
		const code = `
    let ${variableName} = sin(${_freq.variable});
  `;
		return context.emit(variableName, code, OpType.Regular, _freq);
	};

export class TensorGraph {
	protected shaderBuilder: ShaderBuilder = new ShaderBuilder();
	protected device: GPUDevice;
	protected pipeline: GPUComputePipeline | null = null;
	protected bindGroup: GPUBindGroup | null = null;
	private inputData: Map<string, Float32Array> = new Map();

	private inputCounter: number = 0;

	constructor(device: GPUDevice) {
		this.device = device;
	}

	input(data: Float32Array | number[]): Gen {
		const inputName = `input_${this.inputCounter++}`;
		this.inputData.set(
			inputName,
			Array.isArray(data) ? new Float32Array(data) : data,
		);
		return (context: Context) => {
			(context as ShaderBuilder).addInput(inputName);
			return context.emit(`${inputName}[index]`, "", OpType.Regular);
		};
	}

	output(x: Gen): Gen {
		return (context: Context) => {
			const result = context.gen(x);
			const [v] = context.useVariables("output");
			(context as ShaderBuilder).setOutput(v);
			console.log("result output=", result);
			const code = `${v}[index] = ${result.variable};`;
			return context.emit(v, code, OpType.Regular, result);
		};
	}

	compile(graph: Gen) {
		const result = this.shaderBuilder.gen(graph);
    console.log('compile result=', result);
		const shaderCode = this.shaderBuilder.getShaderCode();

		const shaderModule = this.device.createShaderModule({
			code: shaderCode,
		});

		this.pipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: shaderModule,
				entryPoint: "main",
			},
		});
	}

	async run(outputSize: number): Promise<Float32Array> {
		if (!this.pipeline) {
			throw new Error("Graph not compiled. Call compile() first.");
		}

		const buffers: GPUBuffer[] = [];
		const entries: GPUBindGroupEntry[] = [];

		// Create input buffers
		this.inputData.forEach((data, name) => {
			const buffer = this.device.createBuffer({
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.device.queue.writeBuffer(buffer, 0, data);
			buffers.push(buffer);
			entries.push({
				binding: this.shaderBuilder.getBindingIndex(name),
				resource: { buffer },
			});
		});

		// Create output buffer
		const outputBuffer = this.device.createBuffer({
			size: outputSize * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});
		buffers.push(outputBuffer);
		entries.push({
			binding: this.inputData.size,
			resource: { buffer: outputBuffer },
		});

		this.bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries,
		});

		// Create a single command encoder for all operations
		const commandEncoder = this.device.createCommandEncoder();

		// Dispatch compute pass
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);
		passEncoder.dispatchWorkgroups(Math.ceil(outputSize / 64));
		passEncoder.end();

		// Create result buffer
		const resultBuffer = this.device.createBuffer({
			size: outputSize * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		// Copy output to result buffer
		commandEncoder.copyBufferToBuffer(
			outputBuffer,
			0,
			resultBuffer,
			0,
			outputSize * Float32Array.BYTES_PER_ELEMENT,
		);

		// Submit all commands at once
		this.device.queue.submit([commandEncoder.finish()]);

		// Read the result
		await resultBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = resultBuffer.getMappedRange();
		const resultArray = new Float32Array(arrayBuffer.slice(0));
		resultBuffer.unmap();

		// Clean up
		buffers.forEach((buffer) => buffer.destroy());
		resultBuffer.destroy();

		return resultArray;
	}
}
