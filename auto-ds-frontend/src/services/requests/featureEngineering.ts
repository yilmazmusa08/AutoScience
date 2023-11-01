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
  if (payload.column_name) {
    formData.append("column_name", payload.column_name);
  }
  if (payload.column_1) {
    formData.append("column_1", payload.column_1);
  }
  if (payload.column_2) {
    formData.append("column_2", payload.column_2);
  }
  if (payload.n) {
    formData.append("n", payload.n.toString());
  }
  if (payload.Quartile_1) {
    formData.append("Quartile_1", payload.Quartile_1.toString());
  }
  if (payload.Quartile_3) {
    formData.append("Quartile_3", payload.Quartile_3.toString());
  }
  if (payload.transform_type) {
    formData.append("transform_type", payload.transform_type);
  }
  if (payload.scaler_type) {
    formData.append("scaler_type", payload.scaler_type);
  }
  if (payload.value_to_search) {
    formData.append("value_to_search", payload.value_to_search);
  }
  if (payload.value_to_replace) {
    formData.append("value_to_replace", payload.value_to_replace);
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
