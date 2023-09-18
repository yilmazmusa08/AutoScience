import { IPreprocessingRequest } from "../../containers/Preprocessing/types/preprocessing";
import BackendClient from "../BackendClient";
import { PreprocessingUrl } from "../constants/urls";

export function preprocessRequest(
  payload: IPreprocessingRequest,
): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", payload.file);
  return BackendClient.post(PreprocessingUrl.Preprocess, formData)
    .then(({ data }) => {
      return Promise.resolve(data);
    })
    .catch((error) => {
      console.error(error);
      return Promise.reject(error);
    });
}
