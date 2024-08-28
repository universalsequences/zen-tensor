import { TensorGraph } from "./graph";
import { Context } from "./context";
import { Gen, DataType, OpType } from "./zen";

export class Tensor {
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
		this.graph.updateTensor(this.name, data);
		return this;
	}

	fill(value: number) {
		const size = this.shape.reduce((a, b) => a * b, 1);
		return this.set(new Array(size).fill(value));
	}

	ones() {
		return this.fill(1);
	}

	zeroes() {
		return this.fill(0);
	}

	rand() {
		const size = this.shape.reduce((a, b) => a * b, 1);
		const array = new Float32Array(size);
		for (let i = 0; i < size; i++) {
			array[i] = Math.random();
		}
		return this.set(array);
	}

	gen(): Gen {
		return (context: Context) => {
			context.addInput(this.name);
			return {
				...context.emit(this.name, "", OpType.Regular, this.shape),
				type: DataType.Tensor,
			};
		};
	}
}
