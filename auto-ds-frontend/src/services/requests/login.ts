import {
  ILogin,
  IAuthResponse,
} from "../../containers/Authentication/types/authentication";
import BackendClient from "../BackendClient";
import { AuthenticationUrl } from "../constants/urls";

export function loginRequest(payload: ILogin): Promise<IAuthResponse> {
  return BackendClient.post(AuthenticationUrl.Login, payload)
    .then(({ data }) => {
      return Promise.resolve(data);
    })
    .catch((error) => {
      console.error(error);
      return Promise.reject(error);
    });
}
