"use client";
import React, { useEffect, useState, useRef, useCallback } from "react";
import { ArrowRightIcon, ArrowLeftIcon } from "@radix-ui/react-icons";
import Prism from "prismjs";
import "prismjs/themes/prism-okaidia.css"; // Import the PrismJS theme
import "prismjs/components/prism-wgsl"; // Import the specific language syntax

// Import your TensorGraph class and other necessary types
import {
  relu,
  TensorGraph,
  log2,
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
  sigmoid,
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

  const run = useCallback(async () => {
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

      // Network parameters
      const inputSize = 10; // Adjust based on your input data
      const hiddenSize1 = 16;
      const hiddenSize2 = 8;
      const outputSize = 1;
      const batchSize = 32; // Adjust based on your needs

      // Input tensor
      const X = g.tensor([batchSize, inputSize], "X").rand(); // Replace with your actual input data
      const y = g.tensor([batchSize, outputSize], "y").rand().round(); // Replace with your actual labels

      // Layer 1
      const W1 = g.tensor([inputSize, hiddenSize1], "W1").rand().mul(0.01); // Xavier initialization
      const b1 = g.tensor([hiddenSize1], "b1").zeros();
      const layer1 = relu(add(matmul(X, W1), b1));
      /*

      // Layer 2
      const W2 = g.tensor([hiddenSize1, hiddenSize2], "W2").rand().mul(0.01);
      const b2 = g.tensor([hiddenSize2], "b2").zeros();
      const layer2 = relu(add(matmul(layer1, W2), b2));

      // Output layer
      const W3 = g.tensor([hiddenSize2, outputSize], "W3").rand().mul(0.01);
      const b3 = g.tensor([outputSize], "b3").zeros();
      const logits = add(matmul(layer2, W3), b3);
      const predictions = sigmoid(logits);
      */
      const predictions = layer1;

      console.log("created pred");
      // Loss function
      const loss = g.output(binaryCrossEntropy(predictions, y));

      console.log("created loss expression");

      // Compile the graph
      g.compile(loss, [batchSize, inputSize]);

      console.log("compiled");
      // Training loop
      const numEpochs = 100;
      const learningRate = 0.01;

      for (let epoch = 0; epoch < numEpochs; epoch++) {
        const { forward, gradients } = await g.run();

        setGrads(gradients);
        setResult(predictions.node?.result || []);

        // Update weights and biases
        W1.learn(learningRate);
        //b1.learn(learningRate);

        /*
        W2.learn(learningRate);
        b2.learn(learningRate);
        W3.learn(learningRate);
        b3.learn(learningRate);
        */

        const map = new Map<string, Float32Array>();
        map.set("W1", W1.val());
        /*
        map.set("b1", b1.val());
        map.set("W2", W2.val());
        map.set("b2", b2.val());
        map.set("W3", W3.val());
        map.set("b3", b3.val());
        */
        setTensors(map);

        // Log progress (e.g., every 10 epochs)
        if (epoch % 10 === 0) {
          console.log(`Epoch ${epoch}, Loss: ${forward[0]}`);
        }
      }
    } catch (e) {
      console.log("caught error");
      console.log(e);
    }
  }, []);

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
        const si = 4;
        const si2 = 4;

        /*
        const a = g.tensor([si], "a").fill(1);
        const b = g.tensor([si], "b").fill(1);
        const c = g.tensor([si], "c").fill(1);
        const net = mult(a, b);
        */
        const a = g.tensor([si, si], "a").rand();
        const b = g.tensor([si, si], "b").fill(0.92);
        const c = g.tensor([si, si], "c").fill(0.9);
        const d = g.tensor([si], "d").fill(-0.5);
        //const net = add(1, a); // Simply use a single variable
        //const computation_a = sigmoid(relu(add(d, a)));
        const computation = sigmoid(matmul(a, b)); //sigmoid(matmul(add(a, b), b));

        const result = g.output(binaryCrossEntropy(computation, c));
        g.compile(result, [si, si]);

        // update ui
        setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
        setBackwards(g.backpasses);

        for (let i = 0; i < 1000; i++) {
          const { forward, gradients } = await g.run();
          if (computation.node) {
            setComputation(printAST(computation.node));
          }
          setGrads(gradients);

          // setResult(Array.from(forward));
          setEpoch(i);
          a.learn(0.001);
          //b.learn(0.001);
          /*
          console.log("computation.node.result=", computation.node?.result);
          console.log("computation.node.gradient=", computation.node?.gradient);
          console.log("computation_a.node.result=", computation_a.node?.result);
          console.log("computation_a.node.gradient=", computation_a.node?.gradient);
          */
          setResult(computation.node?.result || []);
          const map = new Map<string, Float32Array>();
          map.set("a", a.val());
          map.set("b", b.val());
          setTensors(map);
          //await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } catch (err) {
        console.log(err);
        setError(err instanceof Error ? err.message : "An unknown error occurred");
      }
    }

    //runTensorComputation();
    run();
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
