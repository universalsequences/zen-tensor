enum OpType {
	Regular,
	Reduction,
}

type Gen = (context: Context) => GenResult;
let counter = 0;

enum Type {
	Scalar = 0,
	Tensor = 1,
}

const variable = (x: GenResult, type: Type = Type.Scalar, index?: string) => {
	if (x.type === Type.Tensor && type === Type.Scalar) {
		if (index) {
			return `${x.variable}[${index}]`;
		} else {
			return `${x.variable}[index]`;
		}
	}
	return x.variable;
};

interface GenResult {
	variable: string;
	code: string;
	dependencies: GenResult[];
	opType: OpType;
	context: Context;
	type: Type;
	shape: number[]; // [rows, cols] for 2D, [length] for 1D
}

type Arg = Gen | InputPlaceholder;

interface Context {
	id: number;
	parentContext: Context | undefined;
	gen: (x: Arg) => GenResult;
	useVariables: (...names: string[]) => string[];
	emit: (
		variable: string,
		code: string,
		opType: OpType,
		shape: number[],
		...dependencies: GenResult[]
	) => GenResult;
	useContext: (opType: OpType) => Context;
	getBindingIndex: (name: string) => number;
	addInput: (x: string) => void;
	addOutput: (x: string) => void;
	getShape: (variable: string) => number[];
	setShape: (variable: string, shape: number[]) => void;
}

class KernelContext implements Context {
	private code: string[] = [];
	private inputs: Map<string, number> = new Map();
	private outputs: Map<string, number> = new Map();
	private idx = 0;

	readonly opType: OpType;
	parentContext: Context | undefined;
	id: number;
	tensorGraph: TensorGraph;
	private shapes: Map<string, number[]> = new Map();

	constructor(
		opType: OpType,
		tensorGraph: TensorGraph,
		parentContext?: Context,
	) {
		this.opType = opType;
		this.tensorGraph = tensorGraph;
		this.parentContext = parentContext;
		this.id = counter++;
	}

	getShape(variable: string): number[] {
		return this.shapes.get(variable) || [];
	}

	setShape(variable: string, shape: number[]) {
		this.shapes.set(variable, shape);
	}

	gen(x: Arg): GenResult {
		if (x instanceof InputPlaceholder) {
			const rs = (x as InputPlaceholder).getGen()(this);
			return rs;
		}
		const result = (x as Gen)(this);
		if (result.opType !== this.opType) {
			// We're crossing context boundaries
			const outputName = `cross_context_output_${this.id}_${this.idx++}`;
			console.log(
				"adding output in gen: outputName: ",
				outputName,
				result,
				this,
			);
			this.addInput(outputName);
			const buffer = this.tensorGraph.device.createBuffer({
				size: this.tensorGraph.outputSize * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.tensorGraph.inputBuffers.set(outputName, buffer);

			result.context.addOutput(outputName);
			const code = `${outputName}[index] = ${result.variable};`;
			return {
				context: result.context,
				dependencies: [result],
				opType: result.opType,
				variable: `${outputName}`,
				shape: this.tensorGraph.outputShape,
				code: code,
				type: Type.Tensor,
			};
		}
		return result;
	}

	useVariables(...names: string[]) {
		this.idx++;
		return names.map((n) => `${n}${this.idx}`);
	}

	emit(
		variable: string,
		code: string,
		opType: OpType,
		shape: number[],
		...dependencies: GenResult[]
	): GenResult {
		return {
			context: this,
			variable,
			code,
			shape,
			dependencies,
			opType,
			type: Type.Scalar,
		};
	}

	useContext(opType: OpType): Context {
		if (this.opType !== opType) {
			return new KernelContext(opType, this.tensorGraph, this);
		}
		return this;
	}

	addInput(name: string) {
		if (!this.inputs.has(name)) {
			this.inputs.set(name, this.inputs.size);
		}
	}

	addOutput(name: string) {
		if (!this.outputs.has(name)) {
			this.outputs.set(name, this.outputs.size);
		}
	}

	getBindingIndex(name: string): number {
		if (this.inputs.has(name)) {
			return this.inputs.get(name)!;
		}
		if (this.outputs.has(name)) {
			return this.inputs.size + this.outputs.get(name)!;
		}
		throw new Error(`Binding index not found for ${name}`);
	}

	getShaderCode(): string {
		console.log("get shader code called with codes", this.code, this);
		const inputBindings = Array.from(this.inputs.entries())
			.map(
				([name, index]) =>
					`@group(0) @binding(${index}) var<storage, read> ${name}: array<f32>;`,
			)
			.join("\n");

		const outputBindings = Array.from(this.outputs.entries())
			.map(
				([name, index]) =>
					`@group(0) @binding(${this.inputs.size + index}) var<storage, read_write> ${name}: array<f32>;`,
			)
			.join("\n");

		return `
      ${inputBindings}
      ${outputBindings}

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        ${this.code.join("\n")}
      }
    `;
	}

	getInputs(): string[] {
		return Array.from(this.inputs.keys());
	}

	getOutputs(): string[] {
		return Array.from(this.outputs.keys());
	}
}

class InputPlaceholder {
	private name: string;
	private graph: TensorGraph;
	private shape: number[];

	constructor(name: string, graph: TensorGraph, shape: number[]) {
		this.name = name;
		this.graph = graph;
		this.shape = shape;
	}

	set(data: number[] | Float32Array) {
		const size = this.shape.reduce((a, b) => a * b, 1);
		if (data.length !== size) {
			throw new Error(
				`Input size mismatch. Expected ${size}, got ${data.length}`,
			);
		}
		this.graph.updateInput(this.name, data);
	}

	getGen(): Gen {
		return (context: Context) => {
			context.addInput(this.name);
			context.setShape(this.name, this.shape);
			return {
				...context.emit(this.name, "", OpType.Regular, this.shape),
				type: Type.Tensor,
			};
		};
	}
}

class Kernel {
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

export class TensorGraph {
	device: GPUDevice;
	private contexts: KernelContext[] = [];
	private kernels: Kernel[] = [];
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
		let currentContext = new KernelContext(OpType.Regular, this);
		this.contexts.push(currentContext);
		console.log("compile called...", this.contexts);

		const traverse = (node: GenResult) => {
			console.log(
				"traversing node.code=%s id=%s",
				node.code,
				node.context.id,
				node,
			);
			if (node.context !== currentContext) {
				currentContext = node.context;
				this.contexts = [currentContext, ...this.contexts];
			}
			currentContext.code = [node.code, ...currentContext.code];
			node.dependencies.forEach(traverse);
		};

		const result = graph(currentContext);
		console.log("result graph=", result);
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

		console.log("traversal=", this.contexts);
		// Create kernels
		console.log("input buffers=", this.inputBuffers);
		this.kernels = this.contexts.map(
			(context) => new Kernel(this.device, context, this.inputBuffers),
		);
		console.log("this contexts=", this.contexts, this.kernels);
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

	async run(): Promise<Float32Array> {
		const commandEncoder = this.device.createCommandEncoder();

		for (let i = 0; i < this.kernels.length; i++) {
			const kernel = this.kernels[i];

			// If this is not the first kernel, we need to copy data from the previous kernel
			if (i > 0) {
				const prevKernel = this.kernels[i - 1];
				const prevOutputs = prevKernel.getOutputBuffers();
				const currentInputs = kernel.context.getInputs();

				for (const inputName of currentInputs) {
					if (prevOutputs.has(inputName)) {
						const sourceBuffer = prevOutputs.get(inputName)!;
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

			console.log(`Kernel ${i} inputs:`, kernel.context.getInputs());
			console.log(`Kernel ${i} outputs:`, kernel.getOutputBuffers().keys());
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

	destroy() {
		this.inputBuffers.forEach((buffer) => buffer.destroy());
		this.kernels.forEach((kernel) => {
			kernel.getOutputBuffer("output").destroy();
		});
	}
}

const binaryOp =
	(name: string, op: string) =>
	(x: Arg, y: Arg) =>
	(context: Context): GenResult => {
		const parent = context;
		console.log("binary op called name=", op);
		context = context.useContext(OpType.Regular);
		const _x = context.gen(x);
		const _y = context.gen(y);

		// Get shapes
		const shapeX = _x.shape;
		const shapeY = _y.shape;

		console.log("shapes", _x.shape, _y.shape);

		// Determine output shape
		let outputShape: number[];
		if (arraysEqual(shapeX, shapeY)) {
			outputShape = shapeX;
		} else if (isScalar(shapeX) || isScalar(shapeY)) {
			outputShape = isScalar(shapeX) ? shapeY : shapeX;
		} else {
			console.log("incompatible");
			throw new Error(
				`Incompatible shapes for ${name} operation: ${shapeX} and ${shapeY}`,
			);
		}

		const [variableName] = context.useVariables(`${name}_result`);

		// Generate code with broadcasting if necessary
		let code: string | undefined = undefined;
		if (arraysEqual(shapeX, shapeY)) {
			console.log("equal...", _x, _y);
			code = `let ${variableName} = ${variable(_x)} ${op} ${variable(_y)};`;
		} else if (isScalar(shapeX)) {
			code = `let ${variableName} = ${variable(_x)}[0] ${op} ${variable(_y)};`;
		} else if (isScalar(shapeY)) {
			code = `let ${variableName} = ${variable(_x)} ${op} ${variable(_y)}[0];`;
		}

		console.log("code=", code);
		if (!code) {
			throw new Error("no code");
		}

		console.log("binary op called parent=", parent);
		console.log("binary op called context=", context);
		console.log(code);

		const ret = context.emit(
			variableName,
			code,
			OpType.Regular,
			outputShape,
			_x,
			_y,
		);
		console.log(ret);
		return ret;
	};

// Helper functions
function arraysEqual(a: number[], b: number[]): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (a[i] !== b[i]) return false;
	}
	return true;
}

function isScalar(shape: number[]): boolean {
	return shape.length === 1 && shape[0] === 1;
}

export const add = binaryOp("add", "+");
export const mult = binaryOp("mult", "*");
export const sub = binaryOp("sub", "-");
export const div = binaryOp("div", "/");

export const reduce =
	(op: string) =>
	(x: Arg) =>
	(context: Context): GenResult => {
		console.log("reduce called");
		const reductionContext = context.useContext(OpType.Reduction);
		const _x = reductionContext.gen(x);
		const [variableName] = reductionContext.useVariables(`reduce_result`);
		const code = `
    var ${variableName} = ${_x.variable}[0];
    for (var i = 1u; i < arrayLength(&${_x.variable}); i = i + 1u) {
      ${variableName} = ${variableName} ${op} ${variable(_x, Type.Scalar, "i")};
    }
  `;
		console.log("reduce called with reductionContext", reductionContext);
		return reductionContext.emit(
			variableName,
			code,
			OpType.Reduction,
			_x.shape,
			_x,
		);
	};

export const sum = reduce("+");

export const mean =
	(x: Arg) =>
	(context: Context): GenResult => {
		const reductionContext = context.useContext(OpType.Reduction);
		const _x = reductionContext.gen(x);
		const [sumVariable] = reductionContext.useVariables(`mean_sum`);
		const [countVariable] = reductionContext.useVariables(`mean_count`);
		const [resultVariable] = reductionContext.useVariables(`mean_result`);

		const code = `
    var ${sumVariable} = 0.0;
    var ${countVariable} = 0u;
    for (var i = 0u; i < arrayLength(&${_x.variable}); i = i + 1u) {
      ${sumVariable} = ${sumVariable} + ${_x.variable}[i];
      ${countVariable} = ${countVariable} + 1u;
    }
    let ${resultVariable} = ${sumVariable} / f32(${countVariable});
  `;

		console.log("mean generated=", _x);

		return reductionContext.emit(
			resultVariable,
			code,
			OpType.Reduction,
			_x.shape,
			_x,
		); // Mean always outputs a single value
	};

export const func = (name: string) => {
	return (freq: Arg) => {
		return (context: Context): GenResult => {
			context = context.useContext(OpType.Regular);
			const [variableName] = context.useVariables(`${name}_result`);
			const _freq = context.gen(freq);
			const code = `
    let ${variableName} = ${name}(${_freq.variable});
  `;
			console.log("sines dep=", _freq, code);
			return context.emit(
				variableName,
				code,
				OpType.Regular,
				_freq.shape,
				_freq,
			);
		};
	};
};

export const sine = func("sin");
export const log2 = func("log2");

export const matmul =
	(a: Arg, b: Arg) =>
	(context: Context): GenResult => {
		const _a = context.gen(a);
		const _b = context.gen(b);

		const shapeA = _a.shape;
		const shapeB = _b.shape;

		// Check if shapes are compatible for matrix multiplication
		if (shapeA.length !== 2 || shapeB.length !== 2 || shapeA[1] !== shapeB[0]) {
			throw new Error(
				`Incompatible shapes for matrix multiplication: ${shapeA} and ${shapeB}`,
			);
		}

		const outputShape = [shapeA[0], shapeB[1]];
		const [resultVar, sum, M, N, K, row, col] = context.useVariables(
			"matmul",
			"sum",
			"M",
			"N",
			"K",
			"row",
			"col",
		);

		const code = `
let ${M} = ${shapeA[0]}u;
let ${N} = ${shapeB[1]}u;
let ${K} = ${shapeA[1]}u;

let ${row} = index / ${N};
let ${col} = index % ${N};

var ${sum} = 0.0;
for (var k = 0u; k < ${K}; k = k + 1u) {
let a_idx = ${row} * ${K} + k;
let b_idx = k * ${N} + ${col};
${sum} = ${sum} + ${_a.variable}[a_idx] * ${_b.variable}[b_idx];
    }

let ${resultVar} = ${sum};
  `;

		return context.emit(resultVar, code, OpType.Regular, outputShape, _a, _b);
	};
