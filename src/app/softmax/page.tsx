"use client";

import { TensorGraph, softmax } from "@/lib";
import { useEffect, useState } from "react";

const SoftmaxPage: React.FC = () => {
  const [kernels, setKernels] = useState<string[]>([]);
  useEffect(() => {
    const init = async () => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error("No appropriate GPUAdapter sounds");
      }
      const device = await adapter.requestDevice();
      const g = new TensorGraph(device);

      const input = g.tensor([4, 1], "A").set([1, 2, 3, 4]);
      const s = softmax(input);
      const out = g.output(s);
      g.compile(out, [4, 1]);
      const { forward, gradients } = await g.run();
      if (s.node?.result) {
        const result = await s.node.result();
        console.log("Result = ", result);
      }
      console.log("Forward = ", forward);

      setKernels(g.kernels.map((x) => x.context?.kernelCode || ""));
    };
    init();
  }, []);
  return (
    <div>
      <div>Softmax</div>
      {kernels.map((code, i) => (
        <pre
          style={{ backgroundColor: "#18181b" }}
          key={i}
          className="p-5 text-xs bg-zinc-900 text-zinc-400 m-1 relative "
        >
          <code style={{ fontSize: 11 }} className="language-wgsl">
            {code}
          </code>
          <div className="absolute bottom-2 right-2 text-xs">kernel #{i}</div>
        </pre>
      ))}
    </div>
  );
};

export default SoftmaxPage;
