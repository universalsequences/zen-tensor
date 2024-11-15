"use client";
import { Tree } from "./Tree";
import React, { useEffect, useState, useRef, useCallback } from "react";
import { ArrowRightIcon, ArrowLeftIcon, PlayIcon, PauseIcon } from "@radix-ui/react-icons";
import Prism from "prismjs";
import "prismjs/themes/prism-okaidia.css"; // Import the PrismJS theme
import "prismjs/components/prism-wgsl"; // Import the specific language syntax

// Import your TensorGraph class and other necessary types
import {
  relu,
  TensorGraph,
  add,
  matmul,
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
import { sineLearner } from "@/examples/sine";
import {
  complexTransformer,
  mediumTransformer,
  mediumTransformer2,
  mediumTransformer3,
  sanityCheckTransformer,
  simpleTransformer,
} from "@/examples/transformer";
import {
  batchNormExample,
  testClusteredData,
  testLearnableParameters,
  testNonLinearClassification,
  testSoftmaxBinaryClassification,
  testSoftmaxCrossEntropyMinimal,
  testSubDivSoftmax,
  testTwoSentenceUncertaintyMatmul2D,
} from "@/examples/soft";

const bin = (predictions: number[], targets: Float32Array) => {
  return predictions.map((p, i) => {
    const t = targets[i];
    const p_clipped = Math.max(Math.min(p, 1 - 1e-7), 1e-7);
    return -(t * Math.log(p_clipped) + (1 - t) * Math.log(1 - p_clipped));
  });
};

const TensorPage: React.FC = () => {
  const running = useRef(false);
  const [isPaused, setIsPaused] = useState(false);
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

  const isPausedRef = useRef(isPaused);
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

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

    const epochRunner = simpleTransformer(g);

    setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
    setBackwards(g.backpasses);
    let finalCounter = 0;
    let learningRate = 0.01;
    for (let i = 0; i < 2000; i++) {
      if (isPausedRef.current) {
        await new Promise((resolve) => setTimeout(resolve, 100)); // Small delay to prevent tight loop
        i--; // Stay on same epoch while paused
        continue;
      }

      const a = new Date().getTime();
      const { learningTime, computation, loss, tensors, gradients, predicition } =
        await epochRunner(learningRate);
      const b = new Date().getTime();
      if (computation) {
        setComputation(computation);
      }

      if (i % 2 !== 0) {
        continue;
      }

      if (loss < 0.7 && learningRate > 0.01) {
        //learningRate *= 0.99;
      }

      if (loss < 0.01) {
        finalCounter++;
        if (finalCounter > 2) {
          break;
        }
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
  }, [isPaused]);

  useEffect(() => {
    runAND();
  }, [runAND]);

  const _grads: { [x: string]: Float32Array } = {};
  for (const k of grads.keys()) {
    _grads[k] = grads.get(k)!;
  }

  const _tensors: { [x: string]: Float32Array } = {};
  for (const k of tensors.keys()) {
    _tensors[k] = tensors.get(k)!;
  }

  const copyForwardKernels = () => {
    navigator.clipboard.writeText(kernels.join("\n\n"));
  };

  const copyBackwardKernels = () => {
    navigator.clipboard.writeText(backwards.join("\n\n"));
  };

  const copyGradients = () => {
    const gradientsText = Object.entries(_grads)
      .map(([name, array]) => `${name}:\n${JSON.stringify(Array.from(array), null, 2)}`)
      .join("\n\n");
    navigator.clipboard.writeText(gradientsText);
  };

  return (
    <div className="p-4">
      {error ? (
        <p className="text-red-500">{error}</p>
      ) : result ? (
        <div>
          <div className="flex gap-2">
            <div className="bg-zinc-900 text-zinc-400 p-2 rounded relative relative mb-5 flex">
              <span className="text-purple-500 mr-2">backend:</span>
              webgpu
              <button
                type="button"
                onClick={() => setIsPaused(!isPaused)}
                className="mb-4 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded flex items-center gap-2 ml-5"
              >
                {isPaused ? <PlayIcon /> : <PauseIcon />}
              </button>
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
                <div key={name}>
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
              <button
                type="button"
                onClick={copyGradients}
                className="absolute right-2 top-2 px-2 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 rounded"
              >
                Copy
              </button>
              {Object.keys(_grads).map((name) => (
                <div key={name}>
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
              <div className="text-center text-purple-500 flex flex-col table">
                <div className="flex items-center justify-center gap-2">
                  <div>forward kernels</div>
                  <button
                    type="button"
                    onClick={copyForwardKernels}
                    className="px-2 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 rounded"
                  >
                    Copy
                  </button>
                </div>
              </div>
              {kernels.map((code, i) => (
                <pre
                  style={{ backgroundColor: "#18181b" }}
                  key={`kernel-${i}`}
                  className="p-5 text-xs bg-zinc-900 text-zinc-400 m-1 relative "
                >
                  <code style={{ fontSize: 11 }} className="language-wgsl">
                    {code}
                  </code>
                  <div className="absolute bottom-2 right-2 text-xs">kernel #{i}</div>
                </pre>
              ))}
            </div>
            <div className="mt-2">
              <div className="text-center text-purple-500 flex flex-col table">
                <div className="flex items-center justify-center gap-2">
                  <div>backwards kernels</div>
                  <button
                    type="button"
                    onClick={copyBackwardKernels}
                    className="px-2 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 rounded"
                  >
                    Copy
                  </button>
                </div>
              </div>
              {backwards.map((code, i) => (
                <pre
                  style={{ backgroundColor: "#18181b" }}
                  key={`backward-${i}`}
                  className="p-5 text-xs bg-zinc-900 text-zinc-400 m-1 relative"
                >
                  <code style={{ fontSize: 11 }} className="language-wgsl">
                    {code}
                  </code>
                  <div className="absolute bottom-2 right-2 text-xs">
                    kernel #{kernels.length + i}
                  </div>
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
