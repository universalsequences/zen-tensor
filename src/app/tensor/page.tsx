"use client";
import { Tree } from "./Tree";
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
  mult,
  sub,
  sine,
  div,
  binaryCrossEntropy,
  sigmoid,
  ASTNode,
  leakyRelu,
} from "@/lib/index"; // Adjust the import path as needed
import { printAST } from "@/lib/print";
import { andPredictor } from "@/examples/and";
import { xorPredictor } from "@/examples/xor";
import { digitClassifier } from "@/examples/mnist";
import { circleClassifier } from "@/examples/circle";
import { shapeClassifier } from "@/examples/shape";
import { shapeNoiseClassifier } from "@/examples/shape-noise";
import { stripesClassifier } from "@/examples/spiral";
import { scaleClassifier } from "@/examples/circleBatchNN";

const bin = (predictions: number[], targets: Float32Array) => {
  return predictions.map((p, i) => {
    const t = targets[i];
    const p_clipped = Math.max(Math.min(p, 1 - 1e-7), 1e-7);
    return -(t * Math.log(p_clipped) + (1 - t) * Math.log(1 - p_clipped));
  });
};

const TensorPage: React.FC = () => {
  const running = useRef(false);
  const [result, setResult] = useState<number[] | null>([]);
  const [kernels, setKernels] = useState<string[]>([]);
  const [backwards, setBackwards] = useState<string[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [grads, setGrads] = useState(new Map<string, Float32Array>());
  const [tensors, setTensors] = useState(new Map<string, Float32Array>());
  const [computation, setComputation] = useState<ASTNode | null>(null);

  useEffect(() => {
    Prism.highlightAll();
  }, [kernels]);

  const runa = useCallback(async () => {
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

    // 1. Define the network parameters
    const inputSize = 2;
    const outputSize = 1;
    const batchSize = 4;
    const learningRate = 0.1;
    const epochs = 20000;

    // 2. Initialize tensors
    const X = g.tensor([batchSize, inputSize], "X").set([
      0,
      0, // AND(0, 0) = 0
      0,
      1, // AND(0, 1) = 0
      1,
      0, // AND(1, 0) = 0
      1,
      1, // AND(1, 1) = 1
    ]);

    const Y = g.tensor([batchSize, outputSize], "Y").set([
      0, // AND(0, 0) = 0
      0, // AND(0, 1) = 0
      0, // AND(1, 0) = 0
      1, // AND(1, 1) = 1
    ]);

    // 3. Initialize weights and bias
    const W = g.tensor([inputSize, outputSize], "W").fill(0);
    const b = g.tensor([outputSize], "b").fill(0);

    // 4. Define the network
    const logits = add(matmul(X, W), b);
    const predictions = sigmoid(logits);

    // 5. Define loss function
    const loss = g.output(binaryCrossEntropy(predictions, Y));

    // 6. Compile the computation graph
    g.compile(loss, [batchSize]);
    setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
    setBackwards(g.backpasses);
    if (predictions.node) {
      setComputation(predictions.node);
    }

    // 7. Training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      const { forward, gradients } = await g.run();

      W.learn(learningRate);
      b.learn(learningRate);

      if (epoch % 1 === 0) {
        setGrads(gradients);

        // Log predictions
        if (predictions.node?.result) {
          const pred = await predictions.node.result();
          setResult(pred);
        }

        // Log weights and bias
        // Update weights and biases
        const map = new Map<string, Float32Array>();
        map.set("W", W.val());
        map.set("b", b.val());
        //map.set("W3", W3.val());
        //map.set("b3", b3.val());
        map.set("x", X.val());
        map.set("y", Y.val());
        setTensors(map);

        let sum = forward.reduce((a, b) => a + b, 0);
        setLoss(sum / forward.length);
        setEpoch(epoch);
      }
    }
  }, []);

  const runAND = useCallback(async () => {
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

    const epochRunner = xorPredictor(g);

    setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
    setBackwards(g.backpasses);

    for (let i = 0; i < 20000; i++) {
      const a = new Date().getTime();
      const { learningTime, computation, loss, tensors, gradients, forward, predicition } =
        await epochRunner(0.1);
      const b = new Date().getTime();
      if (computation) {
        setComputation(computation);
      }

      if (i % 2 !== 0) {
        continue;
      }
      console.log("epoch took %s ms learning took %s ms ", b - a, learningTime);
      setTensors(tensors);
      setEpoch(i);
      setLoss(loss);
      if (predicition) {
        console.log(predicition);
        setResult(predicition);
      }
      setGrads(gradients);
    }
  }, []);

  const run = useCallback(async () => {
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
    const inputSize = 2;
    const batchSize = 4;
    const outputSize = 1;
    const hiddenSize = 4; // New hidden layer

    // 2. Initialize tensors
    const X = g.tensor([batchSize, inputSize], "X").set([
      0,
      0, // AND(0, 0) = 0
      0,
      1, // AND(0, 1) = 0
      1,
      0, // AND(1, 0) = 0
      1,
      1, // AND(1, 1) = 1
    ]);

    const Y = g.tensor([batchSize, outputSize], "Y").set([
      0, // AND(0, 0) = 0
      0, // AND(0, 1) = 0
      0, // AND(1, 0) = 0
      1, // AND(1, 1) = 1
    ]);

    function heInit(shape: [number, number]) {
      const fanIn = shape[0];
      return Array(shape[0] * shape[1])
        .fill(0)
        .map(() => (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn));
    }

    const W1 = g.tensor([inputSize, hiddenSize], "W1").set(heInit([inputSize, hiddenSize]));
    const b1 = g.tensor([hiddenSize], "b1").fill(0);
    const W2 = g.tensor([hiddenSize, outputSize], "W2").xavierInit();

    const b2 = g.tensor([outputSize], "b2").fill(0);

    // Two-layer neural network
    const hidden = leakyRelu(add(matmul(X, W1), b1));
    const logits = add(matmul(hidden, W2), b2);
    const predictions = sigmoid(logits);

    // Loss function: Binary Cross-Entropy
    const loss = g.output(binaryCrossEntropy(predictions, Y));

    // Compile the computation graph
    g.compile(loss, [4]);
    setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
    setBackwards(g.backpasses);
    if (predictions.node) {
      setComputation(predictions.node);
    }

    // Training loop
    let learningRate = 0.01;
    for (let i = 0; i < 20; i++) {
      const { forward, gradients } = await g.run();

      W1.learn(learningRate);
      b1.learn(learningRate);
      W2.learn(learningRate);
      b2.learn(learningRate);

      if (i % 10 === 0) {
        setGrads(gradients);
        if (predictions.node?.result) {
          const pred = (await predictions.node?.result()) || [];
          setResult(pred);
          const manualLoss = bin(pred, Y.val());
        }
        const map = new Map<string, Float32Array>();
        map.set("W1", W1.val());
        map.set("b1", b1.val());
        map.set("W2", W2.val());
        map.set("b2", b2.val());
        map.set("labels", Y.val());
        map.set("X", X.val());
        setTensors(map);
        // Update weights and biases using gradient descent

        // Logging or other processing
        let sum = forward.reduce((a, b) => a + b, 0);
        setLoss(sum / forward.length);
        setEpoch(i);
      }
    }
  }, []);

  const run222 = useCallback(async () => {
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
      /*
         const inputSize = 10; // Adjust based on your input data
         const hiddenSize1 = 16;
         const hiddenSize2 = 8;
         const outputSize = 1;
         const batchSize = 32; // Adjust based on your needs

         // Input tensor
         const X = g.tensor([batchSize, inputSize], "X").rand(); // Replace with your actual input data
         const y = g.tensor([batchSize, outputSize], "y").rand().round(); // Replace with your actual labels

         // Layer nt1
         const W1 = g.tensor([inputSize, hiddenSize1], "W1").rand().mul(0.01); // Xavier initialization
         const b1 = g.tensor([hiddenSize1], "b1").zeros();
         const layer1 = relu(add(matmul(X, W1), b1));

         // Layer 2
         const W2 = g.tensor([hiddenSize1, hiddenSize2], "W2").rand().mul(0.01);
         const b2 = g.tensor([hiddenSize2], "b2").zeros();
         const layer2 = relu(add(matmul(layer1, W2), b2));

         // Output layer
         const W3 = g.tensor([hiddenSize2, outputSize], "W3").rand().mul(0.01);
         const b3 = g.tensor([outputSize], "b3").zeros();
         const logits = add(matmul(layer2, W3), b3);
         const predictions = sigmoid(logits);
         //const predictions = layer1;

         // Loss function
         const loss = g.output(binaryCrossEntropy(predictions, y));

         const map = new Map<string, Float32Array>();
         map.set("W1", W1.val());
         map.set("b1", b1.val());
         map.set("W2", W2.val());
         map.set("b2", b2.val());
         map.set("W3", W3.val());
         map.set("b3", b3.val());
         map.set("y", y.val());
         setTensors(map);
       */
      const inputSize = 10; // Adjust based on your input data
      const hiddenSize = 16;
      const outputSize = 1;
      const batchSize = 18; // Adjust based on your needs

      // Input tensor
      const X = g.tensor([batchSize, inputSize], "X").rand(); // Replace with your actual input data
      const y = g.tensor([batchSize, outputSize], "y").rand().round(); // Replace with your actual labels

      // Layer 1
      const W1 = g.tensor([inputSize, hiddenSize], "W1").rand().mul(0.01); // Xavier initialization
      //const b1 = g.tensor([batchSize, hiddenSize], "b1").zeros();
      const b1 = g.tensor([hiddenSize], "b1").fill(0.01);
      const layer1 = relu(add(matmul(X, W1), b1));

      // Output layer
      const W2 = g.tensor([hiddenSize, outputSize], "W2").ones().mul(0.001);
      const b2 = g.tensor([outputSize], "b2").fill(0);
      //const b2 = g.tensor([batchSize, outputSize], "b2").zeros();
      const logits = add(matmul(layer1, W2), b2);
      const predictions = sigmoid(logits);

      // Loss function
      const loss = g.output(binaryCrossEntropy(predictions, y));

      // Compile the graph
      g.compile(loss, [batchSize, outputSize]);

      setComputation(predictions.node!);

      // console.log("layer1", layer1.node);

      // Training loop
      const numEpochs = 20;
      const learningRate = 0.01;

      setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
      setBackwards(g.backpasses);

      for (let epoch = 0; epoch < numEpochs; epoch++) {
        let a = new Date().getTime();
        const { forward, gradients } = await g.run();
        let b = new Date().getTime();

        W1.learn(learningRate);
        b1.learn(learningRate);
        W2.learn(learningRate);
        b2.learn(learningRate);
        if (epoch % 10 !== 0) {
          continue;
        }
        setGrads(gradients);
        if (predictions.node?.result) {
          setResult((await predictions.node?.result()) || []);
        }

        // Update weights and biases
        const map = new Map<string, Float32Array>();
        map.set("W1", W1.val());
        map.set("b1", b1.val());
        map.set("W2", W2.val());
        map.set("b2", b2.val());
        //map.set("W3", W3.val());
        //map.set("b3", b3.val());
        map.set("y", y.val());
        setTensors(map);
        //W3.learn(learningRate);
        //b3.learn(learningRate);

        // Log progress (e.g., every 10 epochs)
        if (epoch % 10 === 0) {
          console.log(`Epoch ${epoch}, Loss: ${forward[0]}`);
        }
        setEpoch(epoch);
        let sum = forward.reduce((a, b) => a + b, 0) / forward.length;
        setLoss(sum);
      }
    } catch (e) {
      console.log("caught error");
      console.log(e);
    }
  }, []);

  useEffect(() => {
    runAND();
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
              {computation && <Tree epoch={epoch} node={computation} />}
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

              <div className="absolute bottom-1 right-2 text-purple-500 text-xs bg-black">
                forward output epoch: {epoch}{" "}
                <span className="text-red-500">loss: {Math.round(10000 * loss) / 10000}</span>
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
              <div className="text-center text-purple-500 flex flex-col w-32">
                <div>forward</div>
                <ArrowRightIcon className="my-auto mx-auto" />
              </div>
              {kernels.map((code, i) => (
                <pre
                  style={{ backgroundColor: "#18181b" }}
                  key={i}
                  className="p-5 text-xs bg-zinc-900 text-zinc-400 m-1 relative "
                >
                  <code style={{ fontSize: 11 }} className="language-wgsl">
                    {code}
                  </code>
                  <div className="absolute bottom-2 right-2 text-xs">kernel #{i + 1}</div>
                </pre>
              ))}
            </div>
            <div className="mt-2">
              <div className="text-center text-purple-500 flex flex-col w-32">
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
