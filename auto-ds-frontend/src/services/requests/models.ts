import { IModels, IModelsRequest } from "../../containers/Models/types/models";
import BackendClient from "../BackendClient";
import { ModelsUrl } from "../constants/urls";

export function modelsRequest(payload: IModelsRequest): Promise<IModels> {
  const formData = new FormData();
  formData.append("file", payload.file);
  if (payload.target_column) {
    formData.append("target_column", payload.target_column);
  }

  return BackendClient.post(ModelsUrl.RunModels, formData)
    .then(({ data }) => {
      return Promise.resolve(data);
    })
    .catch((error) => {
      console.error(error);
      return Promise.reject(error);
    });
}
