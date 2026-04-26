"""Application-level exceptions."""


class AppError(Exception):
    """Base application exception with an HTTP-oriented status code."""

    status_code = 400

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NotFoundError(AppError):
    status_code = 404


class ValidationError(AppError):
    status_code = 422


class ConflictError(AppError):
    status_code = 409


class IntegrationError(AppError):
    status_code = 502

