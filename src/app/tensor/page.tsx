"use client";
import React, { useEffect, useState, useRef } from "react";
import { ArrowRightIcon, ArrowLeftIcon } from "@radix-ui/react-icons";
import Prism from "prismjs";
import "prismjs/themes/prism-okaidia.css"; // Import the PrismJS theme
import "prismjs/components/prism-wgsl"; // Import the specific language syntax

// Import your TensorGraph class and other necessary types
import {
  relu,
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
  binaryCrossEntropy,
} from "@/lib/index"; // Adjust the import path as needed
import { printAST } from "@/lib/print";

const TensorPage: React.FC = () => {
  const running = useRef(false);
  const [result, setResult] = useState<number[] | null>([]);
  const [kernels, setKernels] = useState<string[]>([]);
  const [backwards, setBackwards] = useState<string[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [grads, setGrads] = useState(new Map<string, Float32Array>());
  const [tensors, setTensors] = useState(new Map<string, Float32Array>());
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

        /*
        const a = g.tensor([si], "a").fill(1);
        const b = g.tensor([si], "b").fill(1);
        const c = g.tensor([si], "c").fill(1);
        const net = mult(a, b);
        */
        const a = g.tensor([si], "a").rand();
        const b = g.tensor([si], "b").ones();
        const c = g.tensor([si], "c").fill(0.1);
        const d = g.tensor([si], "d").fill(-0.1);
        //const net = add(1, a); // Simply use a single variable
        const computation_a = relu(add(d, a));
        const computation = mult(b, computation_a);

        const result = g.output(binaryCrossEntropy(computation, c));
        g.compile(result, [si]);

        // update ui
        setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
        setBackwards(g.backpasses);

        for (let i = 0; i < 10; i++) {
          const { forward, gradients } = await g.run();
          if (computation.node) {
            setComputation(printAST(computation.node));
          }
          setGrads(gradients);

          // setResult(Array.from(forward));
          setEpoch(i);
          a.learn(0.001);
          b.learn(0.01);
          const map = new Map<string, Float32Array>();
          /*
          console.log("computation.node.result=", computation.node?.result);
          console.log("computation.node.gradient=", computation.node?.gradient);
          console.log("computation_a.node.result=", computation_a.node?.result);
          console.log("computation_a.node.gradient=", computation_a.node?.gradient);
          */
          setResult(computation.node?.result || []);
          map.set("a", a.val());
          map.set("b", b.val());
          setTensors(map);
          await new Promise((resolve) => setTimeout(resolve, 1000));
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
    _grads[k] = grads.get(k)!;
  }

  const _tensors: { [x: string]: Float32Array } = {};
  for (const k of tensors.keys()) {
    _tensors[k] = tensors.get(k)!;
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
            <div className="bg-zinc-900 text-zinc-400 p-2 rounded relative flex-1 relative  overflow-scroll text-xs">
              {JSON.stringify(result, null, 2)}
              {Object.keys(_tensors).map((name) => (
                <div>
                  <div className="text-purple-500">{name}</div>
                  <div className="text-wrap">
                    {JSON.stringify(Array.from(_tensors[name]), null, 4)}
                  </div>
                </div>
              ))}

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
              <div className="text-center text-purple-500 flex flex-col mx-auto w-32">
                <div>forward</div>
                <ArrowRightIcon className="my-auto mx-auto" />
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
              <div className="text-center text-purple-500 flex flex-col mx-auto w-32">
                <div>backwards</div>
                <ArrowLeftIcon className="my-auto mx-auto" />
              </div>
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
