import {
  ILogin,
  IAuthResponse,
} from "../../containers/Authentication/types/authentication";
import BackendClient from "../BackendClient";
import { AuthenticationUrl } from "../constants/urls";

export function loginRequest(payload: ILogin): Promise<IAuthResponse> {
  return BackendClient.post(AuthenticationUrl.Login, payload)
    .then(({ data }) => {
      return data;
    })
    .catch((error) => {
      console.log(error);
      throw error
    });
}
