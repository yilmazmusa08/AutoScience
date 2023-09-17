export interface IUser {
  pk: number;
  username: string;
  email: string;
  first_name?: string | null;
  last_name?: string | null;
}
