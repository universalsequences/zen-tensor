import { GenResult, Arg, OpType, Gen, Type, variable } from "./zen";
import { InputPlaceholder } from "./input";
import { TensorGraph } from "./graph";

let counter = 1;

export interface Context {
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

export class KernelContext implements Context {
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
		if (result.opType !== this.opType || result.opType === OpType.Reduction) {
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

			const _out = `${outputName}_out`;
			result.context.addOutput(_out);
			const code = `${_out}[index] = ${variable(result)};`;
			return {
				context: result.context,
				dependencies: [result],
				opType: result.opType,
				variable: `${outputName}`,
				shape: result.shape,
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
		if (this.opType !== opType || opType === OpType.Reduction) {
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
