import { ASTNode } from "./zen";

export const printAST = (astNode: ASTNode): string => {
  const op = astNode.operation || "unknown";
  if (!astNode.dependencies.length) {
    return op;
  }
  let value = "";
  if (astNode.result && astNode.gradient) {
    value = ""; //` [${astNode.result[0]},${astNode.gradient[0]}]`;
  }

  return `(${op} ${astNode.dependencies.map((x) => printAST(x)).join(" ")}${value})`;
};
