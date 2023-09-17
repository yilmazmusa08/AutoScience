import { IAnalysis } from "../types/analysis";

export const mockAnalysis: IAnalysis = {
  Results: {
    "Column Roles": {
      id: {
        Role: "id",
      },
      huml: {
        Role: "scalar",
      },
      humw: {
        Role: "scalar",
      },
      ulnal: {
        Role: "scalar",
      },
      ulnaw: {
        Role: "scalar",
      },
      feml: {
        Role: "scalar",
      },
      femw: {
        Role: "scalar",
      },
      tibl: {
        Role: "scalar",
      },
      tibw: {
        Role: "scalar",
      },
      tarl: {
        Role: "scalar",
      },
      tarw: {
        Role: "scalar",
      },
      type: {
        Role: "categoric",
      },
    },
    Warnings: [
      ["id", "Unique Rate : 100.00%"],
      ["huml", "Unique Rate : 97.14%"],
      ["ulnal", "Unique Rate : 95.00%"],
      ["feml", "Unique Rate : 95.95%"],
      ["tibl", "Unique Rate : 96.67%"],
      ["tarl", "Unique Rate : 97.62%"],
    ],
    Distributions: {
      id: "uniform",
      feml: "gauss",
      tibl: "gauss",
      tibw: "gauss",
      type: "beta",
    },
    "Problem Type": ["Clustering"],
    PCA: {
      "Cumulative Explained Variance Ratio": [0.79, 0.96],
      "Principal Component": [1, 2],
    },
  },
};
