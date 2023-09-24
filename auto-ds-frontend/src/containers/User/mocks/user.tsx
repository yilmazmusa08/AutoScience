import { IAuthUser, IUser } from "../types/user";

export const mockAuthUser: IAuthUser = {
  pk: 2,
  username: "test",
  email: "test@test.test",
  first_name: "",
  last_name: "",
};

export const mockUser: IUser = {
  ...mockAuthUser, // Include properties from mockAuthUser
  date_joined: new Date("2023-09-24T00:00:00Z"), // Replace with your desired date format
  last_login: null, // Replace with the last login date if available
};
