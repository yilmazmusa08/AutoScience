import { IFeatureEngineeringRequest } from "../../containers/FeatureEngineering/types/featureEngineering";
import BackendClient from "../BackendClient";
import { FeatureEngineeringUrl } from "../constants/urls";

export function featureEngineeringRequest(
  payload: IFeatureEngineeringRequest,
): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", payload.file);
  formData.append("operation", payload.operation);
  if (payload.method) {
    formData.append("method", payload.method);
  }
  if (payload.col_name) {
    formData.append("col_name", payload.col_name);
  }
  if (payload.col1) {
    formData.append("col1", payload.col1);
  }
  if (payload.col2) {
    formData.append("col2", payload.col2);
  }
  if (payload.n) {
    formData.append("n", payload.n.toString());
  }
  if (payload.Q1) {
    formData.append("Q1", payload.Q1.toString());
  }
  if (payload.Q3) {
    formData.append("Q3", payload.Q3.toString());
  }
  if (payload.transform_type) {
    formData.append("transform_type", payload.transform_type);
  }
  if (payload.scaler_type) {
    formData.append("scaler_type", payload.scaler_type);
  }
  if (payload.val_search) {
    formData.append("val_search", payload.val_search);
  }
  if (payload.val_replace) {
    formData.append("val_replace", payload.val_replace);
  }
  if (payload.remove) {
    formData.append("remove", payload.remove.toString());
  }
  return BackendClient.post(FeatureEngineeringUrl.RunFeatureEngineering, formData)
    .then(({ data }) => {
      return Promise.resolve(data);
    })
    .catch((error) => {
      console.error(error);
      return Promise.reject(error);
    });
}
