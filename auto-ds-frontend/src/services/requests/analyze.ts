import {
  IAnalysis,
  IAnalysisRequest,
} from "../../containers/Analysis/types/analysis";
import BackendClient from "../BackendClient";
import { AnalysisUrl } from "../constants/urls";

export function analyzeRequest(payload: IAnalysisRequest): Promise<IAnalysis> {
  return BackendClient.post(AnalysisUrl.Analyze, payload)
    .then(({ data }) => {
      return data;
    })
    .catch((error) => {
      console.log(error);
      return {} as IAnalysis;
    });
}
