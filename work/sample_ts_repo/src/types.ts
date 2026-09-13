export interface User {
  id: number;
  name: string;
  email: string;
}

export type ID = string | number;

export enum Role {
  Admin,
  Editor,
  Viewer,
}

export interface Session {
  token: string;
  expiresAt: number;
}