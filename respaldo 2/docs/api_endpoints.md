# API Endpoints

## Authentication
- `POST /api/auth/register`: Register a new user.
- `POST /api/auth/login`: Authenticate and receive a token.

## Inquiries
- `GET /api/inquiries`: List all inquiries (User sees their own, Admin sees all).
- `POST /api/inquiries`: Create a new inquiry.
- `GET /api/inquiries/<id>`: Get details of a specific inquiry.
- `PUT /api/inquiries/<id>`: Update an inquiry (e.g., close it).

## Responses
- `POST /api/inquiries/<id>/responses`: Add a response to an inquiry.

## Categories
- `GET /api/categories`: List available categories.

## AI Integration
- `POST /api/ai/analyze`: Send inquiry text to Gemini for preliminary analysis/suggestion.
