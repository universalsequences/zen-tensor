import {
  ASTNode,
  Arg,
  OpType,
  Gen,
  DataType,
  toScalar,
  intermediate,
  intermediateVar,
} from "./zen";
import { Tensor } from "./input";
import { TensorGraph } from "./graph";
import { constructGroup } from "./utils";
import { BackwardContext } from "./back";

let contextIdx = 1;

const MAX_BUFFERS = 4;

export interface BaseContext<T> {
  code: string[];
  opType: OpType;
  emit: (
    variable: string,
    code: string,
    opType: OpType,
    shape: number[],
    ...dependencies: T[]
  ) => T;
  useVariables: (...names: string[]) => string[];
}

export type Context<T> = BaseContext<T> & {
  backward?: BackwardContext;
  useContext: (opType: OpType) => Context<T>;
  usedVariables: string[];
  kernelCode?: string;
  id: number;
  children: Context<T>[];
  parentContext: Context<T> | undefined;
  gen: (x: Arg, force?: boolean) => T;
  addInput: (x: string) => void;
  addOutput: (x: string) => void;
  inputs: Map<string, number>;
  idx: number;
  outputs: Map<string, number>;
  generateKernel: () => string;
  intermediateOutputs: string[];
  getInputs: () => string[];
  getOutputs: () => string[];
  lazyInputs: string[];
  evalLazyInputs: () => void;
};

const visited = new Map<string, ASTNode>();

/**
 * A KernelContext collects operations that may be run on the same kernel
 * Using this, it generates code for the
 * */
export class KernelContext implements Context<ASTNode> {
  code: string[] = [];
  inputs: Map<string, number> = new Map();
  outputs: Map<string, number> = new Map();
  kernelCode?: string;
  opType: OpType;
  parentContext: Context<ASTNode> | undefined;
  id: number;
  tensorGraph: TensorGraph;
  children: Context<ASTNode>[] = [];
  idx = 0;
  usedVariables: string[] = [];
  intermediateOutputs: string[] = [];
  backwardContext?: BackwardContext;
  lazyInputs: string[] = [];

  constructor(opType: OpType, tensorGraph: TensorGraph, parentContext?: Context<ASTNode>) {
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
      // todo need a number op to be used here...
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

    // TODO - can we place this in the context?
    const memoized = visited.get(result.variable);
    if (memoized) {
      this.addInput(memoized.variable);
      return memoized;
    }

    if (result.opType === OpType.Reshape) {
      return result;
    }

    if (!this.usedVariables.includes(result.variable)) {
      console.log("YOOOO USED VAR NOT HAVE", result.variable, result);
      this.lazyInputs.push(result.variable);
      result.variable += "_intermediate";
      result.type = DataType.Tensor;
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
        gradientVariable: result.gradientVariable,
      };
      visited.set(result.variable, x);
      return x;
    }
    return result;
  }

  useVariables(...names: string[]) {
    this.idx++;
    const variables = names.map((n) => `${n}${this.idx}`);
    this.usedVariables.push(...variables);
    return variables;
  }

  emit(
    variable: string,
    code: string,
    opType: OpType,
    shape: number[],
    ...dependencies: ASTNode[]
  ): ASTNode {
    const gradientVariable = `grad_${variable}`;
    let astNode = {
      context: this,
      gradientVariable,
      variable,
      code,
      shape,
      dependencies,
      opType,
      type: DataType.Scalar,
    };
    for (const dep of dependencies) {
      dep.parent = astNode;
    }
    return astNode;
  }

  useContext(opType: OpType): Context<ASTNode> {
    if (this.usedVariables.length + this.inputs.size + this.outputs.size > MAX_BUFFERS) {
      return new KernelContext(opType, this.tensorGraph, this);
    }
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

  generateKernel(useIntermediates = true): string {
    const inputBindings = Array.from(this.inputs.entries())
      .map(([name, index]) => constructGroup(index, "read", name))
      .join("\n");

    let outputBindings = Array.from(this.outputs.entries())
      .map(([name, index]) => constructGroup(this.inputs.size + index, "read_write", name))
      .join("\n");

    const code = this.code
      .filter((x) => x !== "")
      .flatMap((c) => c.split("\n"))
      .map((x) => "        " + x)
      .join("\n");

    const intermediates = !useIntermediates
      ? []
      : this.usedVariables.filter((x) => !x.includes("output") && code.includes(x)); // TODO - use something less hacky

    const intermediateValues = intermediates
      .map(
        (intermediateValue) =>
          `        ${intermediateVar(intermediateValue)}[index] = ${intermediateValue};`,
      )
      .join("\n");

    if (useIntermediates) {
      let i = 0;
      for (const inter of intermediates) {
        outputBindings +=
          "\n" +
          constructGroup(
            this.outputs.size + this.inputs.size + i,
            "read_write",
            intermediateVar(inter),
          ) +
          "\n";

        i++;
      }

      this.intermediateOutputs = intermediates;
    }

    return `
${inputBindings}
${outputBindings}

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
${code}

        // intermediate values
${intermediateValues}
      }
    `;
  }

  getInputs(): string[] {
    return Array.from(this.inputs.keys());
  }

  evalLazyInputs() {
    console.log("LAZY INPUTS=", this.lazyInputs);
    for (const ii of this.lazyInputs) {
      const inp = ii + "_intermediate";
      this.tensorGraph.inputData.set(inp, new Float32Array(this.tensorGraph.outputSize));
      this.addInput(inp);
    }
  }

  getOutputs(): string[] {
    return Array.from(this.outputs.keys());
  }
}
