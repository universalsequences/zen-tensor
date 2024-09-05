import { TensorGraph } from "./graph";
import { Context } from "./context";
import { Gen, DataType, OpType, ASTNode } from "./zen";
import { memo } from "./memo";

export class Tensor {
  private name: string;
  private graph: TensorGraph;
  private shape: number[];
  private i = 0;

  constructor(name: string, graph: TensorGraph, shape: number[]) {
    this.name = name;
    this.graph = graph;
    this.shape = shape;
  }

  set(data: number[] | Float32Array) {
    const size = this.shape.reduce((a, b) => a * b, 1);
    if (data.length !== size) {
      throw new Error(`Input size mismatch. Expected ${size}, got ${data.length}`);
    }
    this.graph.updateTensor(this.name, data);
    return this;
  }

  round() {
    let val = this.val();
    if (val) {
      for (let i = 0; i < val.length; i++) {
        val[i] = Math.round(val[i]);
      }
    }
    return this;
  }

  mul(factor: number) {
    let val = this.val();
    if (val) {
      for (let i = 0; i < val.length; i++) {
        val[i] *= factor;
      }
    }
    return this;
  }

 sub(factor: number) {
    let val = this.val();
    if (val) {
      for (let i = 0; i < val.length; i++) {
        val[i] -= factor;
      }
    }
    return this;
  }



  grad() {
    return this.graph.gradientData.get(this.name)!;
  }

  val() {
    return this.graph.inputData.get(this.name)!;
  }

  learn(learningRate: number) {
    const grad = this.grad();
    const val = this.val();

    const result = new Float32Array(val.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = val[i] - learningRate * grad[i];
    }
    this.set(result);
    if (this.i % 10 === 0) {
      // console.log("tensor. val/grad", this.name, val, grad);
    }
    this.i++;
  }

  fill(value: number) {
    const size = this.shape.reduce((a, b) => a * b, 1);
    return this.set(new Array(size).fill(value));
  }

  ones() {
    return this.fill(1);
  }

  zeros() {
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

  randn() {
    const size = this.shape.reduce((a, b) => a * b, 1);
    const array = new Float32Array(size);

    for (let i = 0; i < size; i += 2) {
      // Box-Muller transform
      const u1 = Math.random();
      const u2 = Math.random();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

      array[i] = z1;
      if (i + 1 < size) {
        array[i + 1] = z2;
      }
    }

    return this.set(array);
  }

  xavierInit() {
    const fanIn = this.shape[0];
    const fanOut = this.shape[1] || this.shape[0]; // Use fanIn if it's a 1D tensor
    const std = Math.sqrt(2 / (fanIn + fanOut));

    const size = this.shape.reduce((a, b) => a * b, 1);
    const array = new Float32Array(size);

    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

      array[i] = z1 * std;
      if (i + 1 < size) {
        array[i + 1] = z2 * std;
      }
    }

    return this.set(array);
  }

  gen(): Gen {
    return memo(
      (context: Context<ASTNode>) => {
        context.addInput(this.name);
        return {
          ...context.emit(
            extractName(this.name) || "tensor",
            this.name,
            "",
            OpType.Regular,
            this.shape,
          ),
          type: DataType.Tensor,
        };
      },
      () => ({ code: "", intermediateVariables: [] }),
    );
  }
}

export const extractName = (name: string) => {
  const parts = name.split("_");
  return parts.length >= 3 ? parts[1] : "";
};
