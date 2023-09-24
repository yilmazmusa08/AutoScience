export interface IAuthUser {
  pk: number;
  username: string;
  email: string;
  first_name?: string | null;
  last_name?: string | null;
}

export interface IUser extends Omit<IAuthUser, 'pk'> {
  date_joined: Date; // You should adjust the type to match your date format
  last_login: Date | null; // You should adjust the type to match your date format
}