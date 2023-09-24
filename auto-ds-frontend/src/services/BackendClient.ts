import axios, { AxiosRequestConfig } from "axios";
import { message } from "antd";

import constants from "../constants";
import { UserStorage, AuthUser } from "./UserStorage";

const backendInstance = axios.create({
  baseURL: constants.BaseUrl,
});

backendInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error("error", error?.message);
    message.error(error?.message);
    return Promise.reject(error?.message);
  },
);

class Backend {
  private getConfig(): AxiosRequestConfig {
    return {
      headers: {
        ...(UserStorage.isAuthenticated() && {
          Authorization: "Bearer " + UserStorage.getUser(AuthUser.AccessToken),
        }),
      },
    };
  }

  public async delete(url: string) {
    return backendInstance.delete(url, this.getConfig());
  }
  public async get(url: string, config?: any) {
    return backendInstance.get(url, { ...this.getConfig(), ...config });
  }
  public async patch(url: string, data?: any) {
    return backendInstance.patch(url, data, this.getConfig());
  }
  public async post(url: string, data?: any, config?: any) {
    return backendInstance.post(url, data, { ...this.getConfig(), ...config });
  }
  public async put(url: string, data?: any) {
    return backendInstance.put(url, data, this.getConfig());
  }
}

const BackendClient = new Backend();

export default BackendClient;
