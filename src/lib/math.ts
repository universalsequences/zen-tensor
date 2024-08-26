import { Context } from "./context";
import { OpType, variable, Gen, GenResult, Arg, Type } from "./zen";

const binaryOp =
	(name: string, op: string) =>
	(x: Arg, y: Arg) =>
	(context: Context): GenResult => {
		const parent = context;
		context = context.useContext(OpType.Regular);
		const _x = context.gen(x);
		const _y = context.gen(y);

		// Get shapes
		const shapeX = _x.shape;
		const shapeY = _y.shape;

		// Determine output shape
		let outputShape: number[];
		if (arraysEqual(shapeX, shapeY)) {
			outputShape = shapeX;
		} else if (isScalar(shapeX) || isScalar(shapeY)) {
			outputShape = isScalar(shapeX) ? shapeY : shapeX;
		} else {
			throw new Error(
				`Incompatible shapes for ${name} operation: ${shapeX} and ${shapeY}`,
			);
		}

		const [variableName] = context.useVariables(`${name}_result`);

		// Generate code with broadcasting if necessary
		let code: string | undefined = undefined;
		if (arraysEqual(shapeX, shapeY)) {
			code = `let ${variableName} = ${variable(_x)} ${op} ${variable(_y)};`;
		} else if (isScalar(shapeX)) {
			code = `let ${variableName} = ${variable(_x)}[0] ${op} ${variable(_y)};`;
		} else if (isScalar(shapeY)) {
			code = `let ${variableName} = ${variable(_x)} ${op} ${variable(_y)}[0];`;
		}

		if (!code) {
			throw new Error("no code");
		}

		return context.emit(
			variableName,
			code,
			OpType.Regular,
			outputShape,
			_x,
			_y,
		);
	};

const binaryOpF =
	(name: string, op: string) =>
	(x: Arg, y: Arg) =>
	(context: Context): GenResult => {
		const parent = context;
		context = context.useContext(OpType.Regular);
		const _x = context.gen(x);
		const _y = context.gen(y);

		// Get shapes
		const shapeX = _x.shape;
		const shapeY = _y.shape;

		// Determine output shape
		let outputShape: number[];
		if (arraysEqual(shapeX, shapeY)) {
			outputShape = shapeX;
		} else if (isScalar(shapeX) || isScalar(shapeY)) {
			outputShape = isScalar(shapeX) ? shapeY : shapeX;
		} else {
			throw new Error(
				`Incompatible shapes for ${name} operation: ${shapeX} and ${shapeY}`,
			);
		}

		const [variableName] = context.useVariables(`${name}_result`);

		// Generate code with broadcasting if necessary
		let code: string;
		const totalSize = outputShape.reduce((a, b) => a * b, 1);

		if (arraysEqual(shapeX, shapeY)) {
			code = `
      for (var index = global_id.x; index < ${totalSize}u; index += workgroup_size.x) {
        ${variableName}[index] = ${_x.variable}[index] ${op} ${_y.variable}[index];
      }
    `;
		} else if (isScalar(shapeX)) {
			code = `
      let x_val = ${_x.variable}[0];
      for (var index = global_id.x; index < ${totalSize}u; index += workgroup_size.x) {
        ${variableName}[index] = x_val ${op} ${_y.variable}[index];
      }
    `;
		} else if (isScalar(shapeY)) {
			code = `
      let y_val = ${_y.variable}[0];
      for (var index = global_id.x; index < ${totalSize}u; index += workgroup_size.x) {
        ${variableName}[index] = ${_x.variable}[index] ${op} y_val;
      }
    `;
		} else {
			throw new Error("Unexpected shape combination");
		}

		return context.emit(
			variableName,
			code,
			OpType.Regular,
			outputShape,
			_x,
			_y,
		);
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
		const reductionContext = context.useContext(OpType.Reduction);
		const _x = reductionContext.gen(x);
		const [variableName] = reductionContext.useVariables(`reduce_result`);
		const code = `
    var ${variableName} = ${_x.variable}[0];
    for (var i = 1u; i < arrayLength(&${_x.variable}); i = i + 1u) {
      ${variableName} = ${variableName} ${op} ${variable(_x, Type.Scalar, "i")};
    }
  `;
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
let ${variableName} = ${name}(${variable(_freq)});
  `;
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
		context = context.useContext(OpType.Reduction);
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
${sum} = ${sum} + ${variable(_a, Type.Scalar, "a_idx")} * ${variable(_b, Type.Scalar, "b_idx")};
    }

let ${resultVar} = ${sum};
  `;

		let m = context.emit(
			resultVar,
			code,
			OpType.Reduction,
			outputShape,
			_a,
			_b,
		);
		return m;
	};
