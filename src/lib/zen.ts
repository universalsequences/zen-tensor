import { Context } from "./context";
import { InputPlaceholder } from "./input";
export enum OpType {
	Regular,
	Reduction,
}

export type Gen = (context: Context) => GenResult;

export enum Type {
	Scalar = 0,
	Tensor = 1,
}

export const variable = (
	x: GenResult,
	type: Type = Type.Scalar,
	index?: string,
) => {
	if (x.type === Type.Tensor && type === Type.Scalar) {
		if (index) {
			return `${x.variable}[${index}]`;
		} else {
			return `${x.variable}[index]`;
		}
	}
	return x.variable;
};

export interface GenResult {
	variable: string;
	code: string;
	dependencies: GenResult[];
	opType: OpType;
	context: Context;
	type: Type;
	shape: number[]; // [rows, cols] for 2D, [length] for 1D
}

export type Arg = Gen | InputPlaceholder;

