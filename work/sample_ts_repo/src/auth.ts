import jwt from "jsonwebtoken";
import { Role, Session, User } from "./types";

/**
 * Handles authentication for the sample service.
 */
export class AuthService {
  private secret = "demo-secret";

  login(username: string, password: string): string | null {
    if (username && password) {
      return jwt.sign({ sub: username }, this.secret);
    }
    return null;
  }

  validateToken(token: string): User {
    return jwt.verify(token, this.secret);
  }
}

export const issueSession = async (user: User): Promise<Session> => {
  const token = jwt.sign({ sub: user.id }, "demo-secret");
  return { token, expiresAt: Date.now() + 3600 };
};

export function requireRole(role: Role): boolean {
  return role === Role.Admin;
}