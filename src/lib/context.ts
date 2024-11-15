import {
  ASTNode,
  Arg,
  OpType,
  Gen,
  DataType,
  toScalar,
  intermediateVar,
  intermediate,
} from "./zen";
import { Tensor } from "./tensor";
import { TensorGraph } from "./graph";
import { constructGroup, shapeToSize } from "./utils";
import { BackwardContext } from "./back";
import { numberOp } from "./number";
import { WORKGROUP_SIZE } from "@/constants/kernel";

let contextIdx = 1;

// DO NOT CHANGE -- WEIGHTS STOP TO UPDATE IF YOU DO. I KNOW THIS LIMITS KERNEL FUSION BUT WILL NEED TO BE WAY SMARTER TO DO THAT
const MAX_BUFFERS = 2;

export interface BaseContext<T> {
  code: string[];
  opType: OpType;
  emit: (
    operation: string,
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
  getAllChildren: () => Context<T>[];
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

  lazyInputShapes: Map<string, number[]>;
  evalLazyInputs: () => void;
  nodes: ASTNode[];
  size: number;
  shape?: number[];
  tensorGraph: TensorGraph;
  getReference: (x: ASTNode) => string;
};

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
  nodes: ASTNode[] = [];
  shape?: number[];
  lazyInputShapes = new Map<string, number[]>();

  constructor(opType: OpType, tensorGraph: TensorGraph, parentContext?: Context<ASTNode>) {
    this.opType = opType;
    this.tensorGraph = tensorGraph;
    this.parentContext = parentContext;
    this.id = contextIdx++;
    if (parentContext) {
      parentContext.children.push(this);
    }
  }

  getReference(node: ASTNode) {
    if (node.type === DataType.Tensor) {
      // grabbing a tensor buffer as is
      return toScalar(node);
    }

    if (node.context !== this) {
      // we are grabbing a reference from a different kernel
      return `${intermediate(node)}[index]`;
    } else {
      return `${node.variable}`;
    }
  }

  getAllChildren() {
    let childs: Context<ASTNode>[] = [];
    for (const c of this.children) {
      childs.push(c);
      childs.push(...c.getAllChildren());
    }
    return childs;
  }

  gen(x: Arg, isReshape?: boolean): ASTNode {
    if (typeof x === "number") {
      return numberOp(x)(this);
    }

    if (x instanceof Tensor) {
      const rs = (x as Tensor).gen()(this);
      return rs;
    }

    if (isReshape) {
      // reshape gen calls to it's arg just get returned w/o any processing.
      return x(this);
    }

    const result = (x as Gen)(this);
    if (result.opType === OpType.Reshape) {
      return result;
    }

    const children = this.getAllChildren();

    if (
      !this.usedVariables.includes(result.variable) &&
      children.some((p) => p.usedVariables.includes(result.variable))
    ) {
      // this tells us that this variable actually exists in a previous context as an intermediate value
      this.lazyInputs.push(result.variable);
      this.lazyInputShapes.set(result.variable, result.shape);
      result.variable += "_intermediate";
      result.type = DataType.Tensor;
    }

    return result;
  }

  useVariables(...names: string[]) {
    let c = this;
    while (c.parentContext) {
      c = c.parentContext;
    }
    c.idx++;
    const variables = names.map((n) => `${n}${c.idx}`);
    this.usedVariables.push(...variables);
    return variables;
  }

  emit(
    operation: string,
    variable: string,
    code: string,
    opType: OpType,
    shape: number[],
    ...dependencies: ASTNode[]
  ): ASTNode {
    const gradientVariable = `grad_${variable}`;
    const astNode = {
      operation,
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
    this.nodes.push(astNode);
    this.shape = shape;
    return astNode;
  }

  useContext(opType: OpType): Context<ASTNode> {
    if (2 * this.usedVariables.length + this.inputs.size /*+ this.outputs.size */ >= MAX_BUFFERS) {
      // we need to ensure that this does not create a backward pass kernel with over 8 buffers
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

    const code = Array.from(new Set(this.code))
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

      @compute @workgroup_size(${WORKGROUP_SIZE})
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
${code}

        // intermediate values
        if (index < ${this.size}) {
${intermediateValues}
  }
      }
    `;
  }

  getInputs(): string[] {
    return Array.from(this.inputs.keys());
  }

  evalLazyInputs() {
    for (const ii of this.lazyInputs) {
      const inp = ii + "_intermediate";
      let found = this.nodes.filter((x) => x.variable === ii);
      let fff = this.lazyInputShapes.get(ii);
      this.tensorGraph.inputData.set(inp, new Float32Array(shapeToSize(fff!)));
      this.addInput(inp);
    }
  }

  get size() {
    return this.shape?.reduce((a, b) => a * b, 1) || this.tensorGraph.outputSize;
  }

  getOutputs(): string[] {
    return Array.from(this.outputs.keys());
  }
}
