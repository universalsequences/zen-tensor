import { Arg, OpType, GenResult, Type } from "./zen";
import { Context } from "./context";

export const reshape =
	(input: Arg, newShape: number[]) =>
	(context: Context): GenResult => {
    context = context.useContext(OpType.Regular)
		const _input = context.gen(input);
		const oldShape = _input.shape;

		// Calculate total elements
		const oldElements = oldShape.reduce((a, b) => a * b, 1);
		const newElements = newShape.reduce((a, b) => a * b, 1);

		if (oldElements !== newElements) {
			throw new Error(
				`Cannot reshape tensor of shape [${oldShape}] to [${newShape}]`,
			);
		}

		// In GPU context, reshape is often just reinterpreting the same data
		// So we don't actually need to generate new shader code for data movement
		const code = "";

		return {
			...context.emit(_input.variable, code, OpType.Regular, newShape, _input),
			type: _input.type,
		};
	};
