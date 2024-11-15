import { trimIndex, v } from "./math";
import { ASTNode } from "./zen";

export const constructGroup = (index: number, type: string, name: string) => {

  return `@group(0) @binding(${index}) var<storage, ${type}> ${name}: array<f32>;`;
};

export const shapeToSize = (x: number[]) => x.reduce((a,b) => a*b, 1);

export const emitIntermediate = (node: ASTNode): string[] => {
  return node.dependencies.map(dep =>
    trimIndex(v(dep)));
}
