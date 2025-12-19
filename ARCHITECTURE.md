# Architecture Overview

## High-Level Design
The application follows a traditional server-rendered architecture with Flask serving HTML templates and providing RESTful JSON APIs for dynamic interactions.

### Frontend (Client-Side)
- **Pure HTML5**: Semantic markup, no JSX or templating frameworks
- **Tailwind CSS**: Utility-first CSS via CDN (no build process required)
- **Vanilla JavaScript**: ES6+ features, no React/Vue/Angular
- **Architecture Pattern**: Progressive Enhancement
  - Server renders initial HTML via Flask templates
  - JavaScript enhances interactivity (AJAX calls, form validation)
  - Works without JavaScript for basic functionality

### Backend (Server-Side)
- **Flask (Python 3.10+)**: Lightweight WSGI web framework
- **Blueprint Architecture**: Modular route organization
- **SQLAlchemy ORM**: Database abstraction layer
- **PostgreSQL**: Relational database for data persistence
- **Google Gemini API**: AI-powered advisory features

## Directory Structure
```
asesoriaimss.io/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── models.py            # SQLAlchemy models
│   ├── routes.py            # Blueprint routes
│   ├── services/            # Business logic (Gemini integration)
│   ├── static/              # CSS, JS, images
│   │   ├── css/
│   │   ├── js/
│   │   └── img/
│   └── templates/           # Jinja2 HTML templates
├── docs/                    # Documentation
├── config.py                # Configuration
└── run.py                   # Application entry point
```

## Data Flow
1. **User Request**: Browser sends HTTP request to Flask server
2. **Routing**: Flask blueprint routes request to appropriate handler
3. **Business Logic**: Handler processes request, may call services (e.g., GeminiService)
4. **Database**: SQLAlchemy queries PostgreSQL via models
5. **Response**: Flask renders Jinja2 template or returns JSON
6. **Client Enhancement**: Vanilla JS handles dynamic interactions via fetch API

## Key Principles
- **No JavaScript Frameworks**: Pure vanilla JS for maximum control and minimal dependencies
- **Server-Side Rendering**: Flask/Jinja2 for initial page loads (SEO-friendly)
- **Progressive Enhancement**: Core functionality works without JS
- **RESTful API**: JSON endpoints for AJAX interactions
