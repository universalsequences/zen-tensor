"use client";

import React, { useEffect, useState, useRef } from "react";

// Import your TensorGraph class and other necessary types
import { TensorGraph, add, reduce, mult, mean, sub, sine } from "@/lib/zen"; // Adjust the import path as needed

const TensorPage: React.FC = () => {
	const running = useRef(false);
	const [result, setResult] = useState<number[][] | null>([]);
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
					throw new Error("No appropriate GPUAdapter found.");
				}

				const device = await adapter.requestDevice();
				const graph = new TensorGraph(device);

				const a = graph.input(8);
				a.set([1, 1, 1, 1, 2, 2, 2, 2]);
				const b = graph.input(8);
				b.set([3, 3, 3, 3, 3, 3, 3, 3]);
				const c = graph.input(1);
				c.set([8]);
				const result = graph.output(mean(mult(a, mult(c, b))));

				graph.compile(result, 8);

				const r = await graph.run();
				setResult((o) => [...o, Array.from(r)]);
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
