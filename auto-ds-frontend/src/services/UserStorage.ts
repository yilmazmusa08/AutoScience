import type { IAuthResponse } from "../containers/Authentication/types/authentication";

export enum AuthUser {
  ID = "user_pk",
  Username = "user_name",
  Email = "user_email",
  AccessToken = "access_token",
  RefreshToken = "refresh_token",
}

export class UserStorage {
  public static isAuthenticated(): boolean {
    return !!this.getUser(AuthUser.AccessToken);
  }

  public static storeUser(auth: IAuthResponse): void {
    localStorage.setItem(AuthUser.ID, auth.user.pk.toString());
    localStorage.setItem(AuthUser.Username, auth.user.username);
    localStorage.setItem(AuthUser.Email, auth.user.email);
    localStorage.setItem(AuthUser.AccessToken, auth.access);
    localStorage.setItem(AuthUser.RefreshToken, auth.refresh);
  }

  public static clear(): void {
    localStorage.removeItem(AuthUser.ID);
    localStorage.removeItem(AuthUser.Username);
    localStorage.removeItem(AuthUser.Email);
    localStorage.removeItem(AuthUser.AccessToken);
    localStorage.removeItem(AuthUser.RefreshToken);
  }

  public static getUser(authUser: AuthUser): string | null {
    try {
      return localStorage.getItem(authUser);
    } catch (err) {
      return null;
    }
  }
}
