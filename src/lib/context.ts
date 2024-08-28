import { ASTNode, Arg, OpType, Gen, DataType, toScalar } from "./zen";
import { Tensor } from "./input";
import { TensorGraph } from "./graph";

let contextIdx = 1;

export interface Context {
  kernelCode?: string;
  id: number;
  children: Context[];
  parentContext: Context | undefined;
  opType: OpType;
  gen: (x: Arg, force?: boolean) => ASTNode;
  useVariables: (...names: string[]) => string[];
  emit: (
    variable: string,
    code: string,
    opType: OpType,
    shape: number[],
    ...dependencies: ASTNode[]
  ) => ASTNode;
  useContext: (opType: OpType) => Context;
  addInput: (x: string) => void;
  addOutput: (x: string) => void;
  inputs: Map<string, number>;
  code: string[];
  idx: number;
  outputs: Map<string, number>;
  generateKernel: () => string;
}

const visited = new Map<string, ASTNode>();
/**
 * A KernelContext collects operations that may be run on the same kernel
 * Using this, it generates code for the
 * */
export class KernelContext implements Context {
  code: string[] = [];
  inputs: Map<string, number> = new Map();
  outputs: Map<string, number> = new Map();
  kernelCode?: string;
  opType: OpType;
  parentContext: Context | undefined;
  id: number;
  tensorGraph: TensorGraph;
  children: Context[] = [];
  idx = 0;

  constructor(opType: OpType, tensorGraph: TensorGraph, parentContext?: Context) {
    this.opType = opType;
    this.tensorGraph = tensorGraph;
    this.parentContext = parentContext;
    this.id = contextIdx++;
    if (parentContext) {
      parentContext.children.push(this);
    }
  }

  gen(x: Arg, force?: boolean): ASTNode {
    if (typeof x === "number") {
      return {
        dependencies: [],
        context: this,
        opType: OpType.Regular,
        type: DataType.Scalar,
        variable: `${x}`,
        code: "",
        shape: [1],
      };
    }
    if (x instanceof Tensor) {
      const rs = (x as Tensor).gen()(this);
      return rs;
    }

    if (force) {
      return x(this);
    }

    const result = (x as Gen)(this);
    const memoized = visited.get(result.variable);
    if (memoized) {
      this.addInput(memoized.variable);
      return memoized;
    }

    if (result.opType === OpType.Reshape) {
      return result;
    }

    if (result.opType !== this.opType || result.opType === OpType.Reduction) {
      // We're crossing context boundaries
      const outputName = `cross_context_output_${this.id}_${this.idx++}`;
      this.addInput(outputName);
      const buffer = this.tensorGraph.device.createBuffer({
        size: this.tensorGraph.outputSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.tensorGraph.inputBuffers.set(outputName, buffer);

      const out = `${outputName}_out`;
      result.context.addOutput(out);
      const code = `${out}[index] = ${toScalar(result)};`;
      let x = {
        context: result.context,
        dependencies: [result],
        opType: result.opType,
        variable: `${outputName}`,
        shape: result.shape,
        code: code,
        type: DataType.Tensor,
      };
      visited.set(result.variable, x);
      return x;
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
    ...dependencies: ASTNode[]
  ): ASTNode {
    return {
      context: this,
      variable,
      code,
      shape,
      dependencies,
      opType,
      type: DataType.Scalar,
    };
  }

  useContext(opType: OpType): Context {
    if (this.opType !== opType || opType === OpType.Reduction) {
      const childrenOfParentWithType =
        this.parentContext?.children.filter((x) => x.opType === opType) || [];
      if (this.opType === opType) {
        childrenOfParentWithType.length = 0;
      }
      if (childrenOfParentWithType.length > 0) {
        return childrenOfParentWithType[0];
      }
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

  generateKernel(): string {
    const inputBindings = Array.from(this.inputs.entries())
      .map(
        ([name, index]) => `@group(0) @binding(${index}) var<storage, read> ${name}: array<f32>;`,
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
${this.code
  .filter((x) => x !== "")
  .flatMap((c) => c.split("\n"))
  .map((x) => "        " + x)
  .join("\n")}
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
