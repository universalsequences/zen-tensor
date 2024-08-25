import { OpType, Arg, Context, GenResult } from "./index";

export const dot =
	(a: Arg, b: Arg) =>
	(context: Context): GenResult => {
    context = context.useContext(OpType.Reduction)
		const _a = context.gen(a);
		const _b = context.gen(b);

		const shapeA = _a.shape;
		const shapeB = _b.shape;

		let outputShape: number[];
		let code: string;

		console.log("dot product 2 ", _a, _b);
		// Determine the type of dot product based on input shapes
		if (shapeA.length === 1 && shapeB.length === 1) {
			// Vector-vector dot product
			if (shapeA[0] !== shapeB[0]) {
				throw new Error(
					`Incompatible shapes for dot product: ${shapeA} and ${shapeB}`,
				);
			}
			outputShape = [1]; // Scalar output
			const [resultVar] = context.useVariables("dot_product");
			code = `
      var ${resultVar} = 0.0;
      for (var i = 0u; i < ${shapeA[0]}u; i = i + 1u) {
        ${resultVar} = ${resultVar} + ${_a.variable}[i] * ${_b.variable}[i];
      }
    `;
			return context.emit(
				resultVar,
				code,
				OpType.Reduction,
				outputShape,
				_a,
				_b,
			);
		} else if (shapeA.length === 2 && shapeB.length === 1) {
			// Matrix-vector dot product
			if (shapeA[1] !== shapeB[0]) {
				throw new Error(
					`Incompatible shapes for matrix-vector dot product: ${shapeA} and ${shapeB}`,
				);
			}
			outputShape = [shapeA[0]]; // Vector output
			const [resultVar] = context.useVariables("dot_product");
			code = `
      for (var i = 0u; i < ${shapeA[0]}u; i = i + 1u) {
        var sum = 0.0;
        for (var j = 0u; j < ${shapeA[1]}u; j = j + 1u) {
          sum = sum + ${_a.variable}[i * ${shapeA[1]}u + j] * ${_b.variable}[j];
        }
        ${resultVar}[i] = sum;
      }
    `;
			return context.emit(
				resultVar,
				code,
				OpType.Reduction,
				outputShape,
				_a,
				_b,
			);
		} else {
			throw new Error(
				`Unsupported shapes for dot product: ${shapeA} and ${shapeB}`,
			);
		}
	};
