import { Context } from "./context";
import { Gen, Generated } from "./gen";

export const op = (name: string, op: string) => {
	return (x: Gen, y: Gen) => {
		return (context: Context): Generated => {
			const _x = context.gen(x);
			const _y = context.gen(y);

			const [variable] = context.useVariables(name);
			const code = `let ${variable} = ${_x.variable} ${op} ${_y.variable}`;
			return {
				code,
				variable,
				dependencies: [_x, _y],
			};
		};
	};
};

export const add = op("add", "+");
export const sub = op("sub", "-");
export const mult = op("mult", "*");
export const div = op("div", "/");
