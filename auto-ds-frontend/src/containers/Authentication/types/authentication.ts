import { IUser } from "../../User/types/user";

export interface ILogin {
  email: string;
  password: string;
}

export interface IRegister {
  email: string;
  username: string;
  password1: string;
  password2: string;
}

export interface IAuthResponse {
  access: string;
  refresh: string;
  user: IUser;
}