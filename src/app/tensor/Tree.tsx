import { ASTNode } from "@/lib";
import { useCallback, useEffect, useState } from "react";

interface Props {
  node: ASTNode;
  epoch: number;
}
export const Tree = (props: Props) => {
  const astNode = props.node;
  const op = astNode.operation || "unknown";
  const [result, setResult] = useState<number[] | null>(null);
  const [gradient, setGradient] = useState<number[] | null>(null);
  const [opened, setOpened] = useState(false);

  useEffect(() => {
    if (!opened) {
      return;
    }
    if (props.node.result) {
      props.node.result().then(setResult);
    }
    if (props.node.gradient) {
      setGradient(props.node.gradient);
    }
  }, [opened, props.epoch, props.node]);

  const onMouseOver = useCallback(async () => {
    if (props.node.result) {
      const r = await props.node.result();
      setResult(r);
      if (props.node.gradient) {
        setGradient(props.node.gradient);
      }
    }
    setOpened(true);
  }, [props.node]);

  let begin = "(";
  let end = ")";
  let opClass = "text-zinc-100";
  if (!astNode.dependencies.length) {
    begin = "";
    end = "";
    opClass = "text-zinc-400";
  }
  return (
    <div className={"flex gap-2" + (opened ? " bg-zinc-800" : "")}>
      {begin}
      <div
        onMouseOver={onMouseOver}
        onMouseLeave={() => {
          setResult(null);
          setOpened(false);
        }}
        className={"relative cursor-pointer"}
      >
        <span className={opened ? "text-red-500" : opClass}>{op}</span>
        {opened && (
          <div
            style={{
              backgroundColor: "#afafaf3f",
              backdropFilter: "blur(8px)",
            }}
            className="absolute -bottom-25 text-xs p-2 text-white z-30 flex max-h-96 rounded-lg transition-all"
          >
            <div className="w-96 overflow-scroll max-h-96">
              <div className="flex gap-2 text-base">
                <div className="text-purple-500 mb-2">{props.node.variable}</div>
                <div className="text-zinc-500">{"///"}</div>
                <div className="text-purple-500 mb-2">{props.node.gradientVariable}</div>
              </div>
              <div className="flex gap-2 text-base">
                <div className="text-purple-500 mb-5">shape: </div>
                <div className="text-purple-500 mb-5 text-purple-100">
                  {JSON.stringify(props.node.shape)}
                </div>
              </div>

              {result && JSON.stringify(result, null, 4)}
            </div>
            {gradient && (
              <div className="w-96 max-h-96 overflow-y-scroll">
                <div className="text-purple-500 text-base">gradient</div>
                {JSON.stringify(gradient, null, 4)}
              </div>
            )}
          </div>
        )}
      </div>
      {"  "}
      {astNode.dependencies.map((x) => (
        <Tree epoch={props.epoch} node={x} />
      ))}{" "}
      {end}
    </div>
  );
};
