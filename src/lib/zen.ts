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
	private variables: Set<string> = new Set();
	private code: string[] = [];
	private inputs: Set<string> = new Set();
	private outputs: Set<string> = new Set();
	private currentOpType: OpType = OpType.Regular;
	private bindingIndices: Map<string, number> = new Map();
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
		this.inputs.add(name);
		this.bindingIndices.set(name, this.bindingIndices.size);
	}

	setOutput(name: string) {
		this.outputs.add(name);
		this.bindingIndices.set(name, this.bindingIndices.size);
	}

	getBindingIndex(name: string): number {
		const index = this.bindingIndices.get(name);
		if (index === undefined) {
			throw new Error(`Binding index not found for ${name}`);
		}
		return index;
	}

	getShaderCode(): string {
		const inputBindings = Array.from(this.inputs)
			.map(
				(name, index) =>
					`@group(0) @binding(${index}) var<storage, read> ${name}: array<f32>;`,
			)
			.join("\n");

		const outputBindings = Array.from(this.outputs)
			.map(
				(name, index) =>
					`@group(0) @binding(${this.inputs.size + index}) var<storage, read_write> ${name}: array<f32>;`,
			)
			.join("\n");

		const stateBinding = `@group(0) @binding(${this.inputs.size + this.outputs.size}) var<storage, read_write> filterState: array<f32>;`;

		console.log("input bindings");
		console.log(inputBindings);

		console.log("output bindings");
		console.log(outputBindings, this.outputs);

		console.log("state bindings");
		console.log(stateBinding);
		return `
      ${inputBindings}
      ${outputBindings}
      ${stateBinding}

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

	constructor(device: GPUDevice) {
		this.device = device;
	}

	input(name: string): Gen {
		return (context: Context) => {
			(context as ShaderBuilder).addInput(name);
			return context.emit(
				`${name}[index]`,
				"",
				OpType.Regular /* no dependencies */,
			);
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
	}

	async run(
		inputs: Record<string, Float32Array>,
		outputSize: number,
	): Promise<Float32Array> {
		if (!this.pipeline) {
			throw new Error("Graph not compiled. Call compile() first.");
		}

		const buffers: GPUBuffer[] = [];
		const entries: GPUBindGroupEntry[] = [];

		// Create input buffers
		Object.entries(inputs).forEach(([name, data], index) => {
			const buffer = this.device.createBuffer({
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.device.queue.writeBuffer(buffer, 0, data);
			buffers.push(buffer);
			entries.push({
				binding: index,
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
			binding: Object.keys(inputs).length,
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
