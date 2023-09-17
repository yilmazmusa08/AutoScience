import {
  IModels,
  IModelsRequest,
} from "../../containers/Models/types/models";
import BackendClient from "../BackendClient";
import { ModelsUrl } from "../constants/urls";

export function modelsRequest(payload: IModelsRequest): Promise<IModels> {
  return BackendClient.post(ModelsUrl.RunModels, payload)
    .then(({ data }) => {
      return data;
    })
    .catch((error) => {
      console.log(error);
      return {} as IModels;
    });
}
