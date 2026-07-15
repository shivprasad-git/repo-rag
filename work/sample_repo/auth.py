import jwt

DEMO_SIGNING_KEY = "demo-signing-key"


class AuthService:
    def validate_token(self, token):
        payload = jwt.decode(token, DEMO_SIGNING_KEY, algorithms=["HS256"])
        return payload["sub"]

    def login(self, username, password):
        if username and password:
            return jwt.encode({"sub": username}, DEMO_SIGNING_KEY, algorithm="HS256")
        return None
