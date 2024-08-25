import { TensorGraph } from "./graph";
import { Context } from "./context";
import { Gen, Type, OpType } from "./zen";

export class InputPlaceholder {
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
		return this;
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
