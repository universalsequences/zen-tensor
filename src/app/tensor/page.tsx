"use client";

import React, { useEffect, useState, useRef } from "react";

// Import your TensorGraph class and other necessary types
import { TensorGraph, add, mult, sub, sine } from "@/lib/zen"; // Adjust the import path as needed

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

				const a = graph.input(6);
				const b = graph.input(6);
				const c = graph.input(1);
				const result = graph.output(
					add(a.getGen(), add(b.getGen(), c.getGen())),
				);

				graph.compile(result, 6);

				let _a = new Float32Array([2, 4, 6, 7, 8, 9]);
				const iterations = 10000;

				for (let i = 0; i < iterations; i++) {
					a.set(_a);
					b.set([1, 2, 3, 4, 5, 6]);
					c.set([3]);

					_a = await graph.run();
					if (i % 100 === 0) console.log("i=%s", i, Array.from(_a));
					setResult((o) => [...o, Array.from(_a)]);
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
