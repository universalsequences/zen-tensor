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
  const [kernels, setKernels] = useState<string[]>([]);
  const [epoch, setEpoch] = useState(0);
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
        const si =4*4;

        const a = g.tensor([si]).rand();
        const b = g.tensor([si]).rand();
        const dim = [Math.sqrt(si), Math.sqrt(si)];
        const result = g.output(matmul(reshape(b, dim), reshape(a, dim)));

        g.compile(result, [si]);

        setKernels(g.kernels.map((x) => x.context.kernelCode || ""));

        for (let i = 0; i < 1; i++) {
          const r = await g.run();
          a.set(r);
          setResult(Array.from(r));
          setEpoch(i);
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } catch (err) {
        console.log(err);
        setError(err instanceof Error ? err.message : "An unknown error occurred");
      }
    }

    runTensorComputation();
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">WebGPU Tensor Computation</h1>
      hello world
      {error ? (
        <p className="text-red-500">{error}</p>
      ) : result ? (
        <div>
          <h2 className="text-xl font-semibold mb-2">Computation Result:</h2>
          epoch: {epoch}
          <pre className="bg-gray-100 p-2 rounded text-zinc-900 relative">
            {JSON.stringify(result, null, 2)}
          </pre>
          <p className="mt-2">
            {kernels.map((k) => (
              <pre className="text-xs bg-zinc-900 text-zinc-400 m-10">{k}</pre>
            ))}
          </p>
        </div>
      ) : (
        <p>Computing...</p>
      )}
    </div>
  );
};

export default TensorPage;
