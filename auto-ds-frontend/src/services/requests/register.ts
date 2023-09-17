import {
  IRegister,
  IAuthResponse,
} from "../../containers/Authentication/types/authentication";
import BackendClient from "../BackendClient";
import { AuthenticationUrl } from "../constants/urls";

export function registerRequest(payload: IRegister): Promise<IAuthResponse> {
  return BackendClient.post(AuthenticationUrl.Register, payload)
    .then(({ data }) => {
      return data;
    })
    .catch((error) => {
      console.log(error);
      throw error
    });
}
