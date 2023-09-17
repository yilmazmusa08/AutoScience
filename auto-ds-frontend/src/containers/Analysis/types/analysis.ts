export interface IAnalysis {
  Results: IAnalysisResults;
}

export interface IAnalysisResults {
  "Column Roles": any;
  Warnings: any;
  Distributions: any;
  "Problem Type": string[];
  PCA: IPca;
}

export interface IPca {
  "Cumulative Explained Variance Ratio": number[];
  "Principal Component": number[];
}

export interface IAnalysisRequest {
  file: File;
  target_column?: string | null;
}
