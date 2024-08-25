import { Gen } from "./gen";

export class Context {
	idx: number;

	constructor() {
		this.idx = 0;
	}

	gen(g: Gen) {
		return g(this);
	}

	useVariables(...names: string[]) {
		this.idx++;
		return names.map((n) => `${n}${this.idx}`);
	}
}
