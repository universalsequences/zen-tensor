"use client";
import React, { useEffect, useState, useRef } from "react";
import Prism from "prismjs";
import "prismjs/themes/prism-okaidia.css"; // Import the PrismJS theme
import "prismjs/components/prism-wgsl"; // Import the specific language syntax

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
  div,
} from "@/lib/index"; // Adjust the import path as needed

const TensorPage: React.FC = () => {
  const running = useRef(false);
  const [result, setResult] = useState<number[] | null>([]);
  const [kernels, setKernels] = useState<string[]>([]);
  const [backwards, setBackwards] = useState<string[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [grads, setGrads] = useState(new Map<string, Float32Array>());
  const [computation, setComputation] = useState("");

  useEffect(() => {
    Prism.highlightAll();
  }, [kernels]);

  useEffect(() => {
    async function runTensorComputation() {
      try {
        if (running.current) return;
        running.current = true;
        if (!navigator.gpu) {
          throw new Error("WebGPU not supported on this browser.");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error("No appropriate GPUAdapter sounds");
        }

        const device = await adapter.requestDevice();
        const g = new TensorGraph(device);
        const si = 3 * 3;

        const a = g.tensor([si]).ones();
        const b = g.tensor([si]).ones();
        const c = g.tensor([si]).ones();
        const d = g.tensor([si]).ones();
        const e = g.tensor([si]).fill(4);
        const dim = [Math.sqrt(si), Math.sqrt(si)];
        const m = matmul(reshape(b, dim), reshape(a, dim));
        //const result = g.output(add(a, mult(c, sum(add(d, b)))));
        //const result = g.output(log2(mult(c, add(a, mult(add(a, b), mult(c, d))))));
        //const result = g.output(mult(b, c));
        //const result = g.output(sum(div(e, add(d, mult(a, add(b, c))))));
        //const result = g.output(mult(e, mult(e, add(a, add(c, b)))));
        const result = g.output(mult(e, sum(add(d, add(a, mult(b, c))))));
        setComputation("mult(e, sum(add(d, add(a, mult(b, c))))))");

        g.compile(result, [si]);

        setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
        setBackwards(g.backpasses);

        for (let i = 0; i < 1; i++) {
          const { forward, gradients } = await g.run();
          setGrads(gradients);
          //b.set(forward);
          setResult(Array.from(forward));
          setEpoch(i);
          //await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } catch (err) {
        console.log(err);
        setError(err instanceof Error ? err.message : "An unknown error occurred");
      }
    }

    runTensorComputation();
  }, []);

  const _grads: { [x: string]: Float32Array } = {};
  for (const k of grads.keys()) {
    _grads[k] = grads.get(k);
  }

  return (
    <div className="p-4">
      {error ? (
        <p className="text-red-500">{error}</p>
      ) : result ? (
        <div>
          <div className="flex gap-2">
            <div className="bg-zinc-900 text-zinc-400 p-2 rounded relative relative mb-5">
              <span className="text-purple-500 mr-2">backend:</span>
              webgpu
            </div>
            <div className="bg-zinc-900 text-zinc-400 p-2 rounded relative flex-1 relative mb-5 flex-1">
              <span className="text-purple-500 mr-2">compute:</span>
              {computation}
            </div>
          </div>
          <div className="flex gap-5 h-64">
            <div className="bg-zinc-900 text-zinc-400 p-2 rounded relative flex-1 relative  overflow-scroll">
              {JSON.stringify(result, null, 2)}

              <div className="absolute bottom-1 right-2 text-purple-500 text-xs">
                forward output epoch: {epoch}
              </div>
            </div>
            <div className="bg-zinc-900 text-zinc-400 text-xs rounded p-2 relative overflow-scroll">
              <div className="text-xs absolute right-5 bottom-2 text-purple-500">gradients</div>
              {Object.keys(_grads).map((name) => (
                <div>
                  <div className="text-purple-500">{name}</div>
                  <div className="w-96 text-wrap">
                    {JSON.stringify(Array.from(_grads[name]), null, 4)}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex pt-5 border-t-zinc-800  gap-5">
            <div className="mt-2">
              <div className="text-center text-zinc-400">
                <div className="text-purple-500">forwards</div>
              </div>
              {kernels.map((code, i) => (
                <pre
                  style={{ backgroundColor: "#18181b" }}
                  key={i}
                  className="p-5 text-xs bg-zinc-900 text-zinc-400 m-1 "
                >
                  <code style={{ fontSize: 11 }} className="language-wgsl">
                    {code}
                  </code>
                </pre>
              ))}
            </div>
            <div className="mt-2">
              <p className="text-center text-purple-500">backwards</p>
              {backwards.map((code, i) => (
                <pre
                  style={{ backgroundColor: "#18181b" }}
                  key={i}
                  className="p-5 text-xs bg-zinc-900 text-zinc-400 m-1"
                >
                  <code style={{ fontSize: 11 }} className="language-wgsl">
                    {code}
                  </code>
                </pre>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p>Computing...</p>
      )}
    </div>
  );
};

export default TensorPage;
