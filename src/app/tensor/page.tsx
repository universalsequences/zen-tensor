"use client";

import React, { useEffect, useState } from "react";

// Import your TensorGraph class and other necessary types
import { TensorGraph, add, mult, sub, sine } from "@/lib/zen"; // Adjust the import path as needed

const TensorPage: React.FC = () => {
	const [result, setResult] = useState<number[] | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		async function runTensorComputation() {
			try {
				if (!navigator.gpu) {
					throw new Error("WebGPU not supported on this browser.");
				}

				const adapter = await navigator.gpu.requestAdapter();
				if (!adapter) {
					throw new Error("No appropriate GPUAdapter found.");
				}

				const device = await adapter.requestDevice();

				const graph = new TensorGraph(device);

				let _a = new Float32Array([20, 40, 60, 70, 80, 90]);
				for (let i = 0; i < 1; i++) {
					const a = graph.input(_a);
					const b = graph.input([1, 2, 3, 4, 5, 6]);
					const c = graph.input([300]);

					// Define matrix multiplication operation
					const result = graph.output(add(sine(sub(mult(a, b), c)), c));

					// Compile the graph
					graph.compile(result);

					// Prepare input data
					_a = await graph.run(6);
					setResult(Array.from(_a));
				}
			} catch (err) {
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
