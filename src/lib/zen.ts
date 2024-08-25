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

class InputPlaceholder {
	private name: string;
	private graph: TensorGraph;
	private size: number;

	constructor(name: string, graph: TensorGraph, size: number) {
		this.name = name;
		this.graph = graph;
		this.size = size;
	}

	set(data: number[] | Float32Array) {
		if (data.length !== this.size) {
			throw new Error(
				`Input size mismatch. Expected ${this.size}, got ${data.length}`,
			);
		}
		this.graph.updateInput(this.name, data);
	}

	getGen(): Gen {
		return (context: Context) => {
			(context as ShaderBuilder).addInput(this.name);
			return context.emit(`${this.name}[index]`, "", OpType.Regular);
		};
	}
}

export class TensorGraph {
  protected shaderBuilder: ShaderBuilder = new ShaderBuilder();
	protected device: GPUDevice;
	protected pipeline: GPUComputePipeline | null = null;
	protected bindGroup: GPUBindGroup | null = null;
	private inputData: Map<string, Float32Array> = new Map();
	private inputBuffers: Map<string, GPUBuffer> = new Map();
	private inputCounter: number = 0;
	private outputSize: number = 0;
	private outputBuffer: GPUBuffer | null = null;

	constructor(device: GPUDevice) {
		this.device = device;
	}

	input(size: number): InputPlaceholder {
		const inputName = `input_${this.inputCounter++}`;
		const placeholder = new InputPlaceholder(inputName, this, size);
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

	output(x: Gen): Gen {
		return (context: Context) => {
			const result = context.gen(x);
			const [v] = context.useVariables("output");
			(context as ShaderBuilder).setOutput(v);
			const code = `${v}[index] = ${result.variable};`;
			return context.emit(v, code, OpType.Regular, result);
		};
	}

	compile(graph: Gen, outputSize: number) {
		this.outputSize = outputSize;
		const result = this.shaderBuilder.gen(graph);
		const shaderCode = this.shaderBuilder.getShaderCode();
		console.log("Generated Shader Code:", shaderCode);

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

		this.setupBuffers();
	}

	private setupBuffers() {
		const entries: GPUBindGroupEntry[] = [];

		// Create input buffers
		this.inputData.forEach((data, name) => {
			const buffer = this.device.createBuffer({
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.device.queue.writeBuffer(buffer, 0, data);
			this.inputBuffers.set(name, buffer);
			entries.push({
				binding: this.shaderBuilder.getBindingIndex(name),
				resource: { buffer },
			});
		});

		// Create output buffer
		this.outputBuffer = this.device.createBuffer({
			size: this.outputSize * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});
		entries.push({
			binding: this.inputData.size,
			resource: { buffer: this.outputBuffer },
		});

		this.bindGroup = this.device.createBindGroup({
			layout: this.pipeline!.getBindGroupLayout(0),
			entries,
		});
	}

	async run(): Promise<Float32Array> {
		if (!this.pipeline || !this.bindGroup || !this.outputBuffer) {
			throw new Error("Graph not compiled. Call compile() first.");
		}

		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);
		passEncoder.dispatchWorkgroups(Math.ceil(this.outputSize / 64));
		passEncoder.end();

		const resultBuffer = this.device.createBuffer({
			size: this.outputSize * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		commandEncoder.copyBufferToBuffer(
			this.outputBuffer,
			0,
			resultBuffer,
			0,
			this.outputSize * Float32Array.BYTES_PER_ELEMENT,
		);

		this.device.queue.submit([commandEncoder.finish()]);

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = resultBuffer.getMappedRange();
		const resultArray = new Float32Array(arrayBuffer.slice(0));
		resultBuffer.unmap();

		resultBuffer.destroy();

		return resultArray;
	}

	destroy() {
		this.inputBuffers.forEach((buffer) => buffer.destroy());
		if (this.outputBuffer) this.outputBuffer.destroy();
	}
}
