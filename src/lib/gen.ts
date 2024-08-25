import { Context } from "./context";
export type Generated = {
	code: string;
	variable: string;
  dependencies: Generated[]
};

export type Gen = (context: Context) => Generated;
