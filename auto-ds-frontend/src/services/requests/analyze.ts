import {
  IAnalysis,
  IAnalysisRequest,
} from "../../containers/Analysis/types/analysis";
import BackendClient from "../BackendClient";
import { AnalysisUrl } from "../constants/urls";

export function analyzeRequest(payload: IAnalysisRequest): Promise<IAnalysis> {
  const formData = new FormData();
  formData.append("file", payload.file);
  if (payload.target_column) {
    formData.append("target_column", payload.target_column);
  }

  return BackendClient.post(AnalysisUrl.Analyze, formData)
    .then(({ data }) => {
      return data;
    })
    .catch((error) => {
      console.log(error);
      return {} as IAnalysis;
    });
}
