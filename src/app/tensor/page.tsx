"use client";

import React, { useEffect, useState, useRef } from "react";

// Import your TensorGraph class and other necessary types
import {
	TensorGraph,
	log2,
	dot,
	reshape,
	add,
	matmul,
	reduce,
	mult,
	mean,
	sub,
	sine,
	sum,
} from "@/lib/index"; // Adjust the import path as needed

const TensorPage: React.FC = () => {
	const running = useRef(false);
	const [result, setResult] = useState<number[] | null>([]);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		async function runTensorComputation() {
			try {
				if (running.current) return;
				running.current = true;
				console.log("run tensor ");
				if (!navigator.gpu) {
					throw new Error("WebGPU not supported on this browser.");
				}

				const adapter = await navigator.gpu.requestAdapter();
				if (!adapter) {
					throw new Error("No appropriate GPUAdapter sounds");
				}

				const device = await adapter.requestDevice();
				const g = new TensorGraph(device);

				const a = g.input([2, 2]).set([1, 4, 1, 1]);
				const b = g.input([2, 2]).set([0.009, 7, 2, 0.4]);
				const c = g.input([2, 2]).set([1, 1, 1, 0.001]);
				const e = g.input([2, 2]).set([1, 1, 1, 1]);
				const result2 = g.output(
					dot(reshape(matmul(a, b), [4]), reshape(matmul(e, b), [4])),
				);
				//const result2 = g.output(mult(matmul(a, b), matmul(e, b)));
				const result6 = g.output(matmul(c, e));
				const result5 = g.output(matmul(a, b));
				//const result = g.output(matmul(a, b));
				//const d = g.input([4]).set([0, 4, 4, 4]);

				// this is complicated because
				const result = g.output(
					sine(
						matmul(
							a,
							mult(
								g.input([2, 2]).set([0.01, -0.1, 0.7, 4.001]),
								add(
									matmul(c, e),
									matmul(
										reshape(matmul(a, b), [2, 2]),
										sum(
											add(
												reshape(add(b, c), [2, 2]),
												reshape(add(e, c), [2, 2]),
											),
										),
									),
								),
							),
						),
					),
				);

				g.compile(result, [2, 2]);

				for (let i = 0; i < 1000; i++) {
					const r = await g.run();
					a.set(r);
					if (i % 100 === 0) {
						console.log("result[%s]", i, Array.from(r));
						setResult(Array.from(r));
					}
				}
			} catch (err) {
				console.log(err);
				setError(
					err instanceof Error ? err.message : "An unknown error occurred",
				);
			}
		}

		runTensorComputation();
	}, []);

	return (
		<div className="p-4">
			<h1 className="text-2xl font-bold mb-4">WebGPU Tensor Computation</h1>
			{error ? (
				<p className="text-red-500">{error}</p>
			) : result ? (
				<div>
					<h2 className="text-xl font-semibold mb-2">Computation Result:</h2>
					<pre className="bg-gray-100 p-2 rounded text-zinc-900">
						{JSON.stringify(result, null, 2)}
					</pre>
					<p className="mt-2">
						This is the result of multiplying a 2x3 matrix by a 3x2 matrix,
						resulting in a 2x2 matrix.
					</p>
				</div>
			) : (
				<p>Computing...</p>
			)}
		</div>
	);
};

export default TensorPage;
