-- Complete Database Schema for asesoriaimss.io

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user', -- 'user', 'professional', 'admin'
    google_id VARCHAR(255) UNIQUE,
    facebook_id VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Professionals table
CREATE TABLE professionals (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    specialty VARCHAR(100),
    city VARCHAR(100),
    bio TEXT,
    rating FLOAT DEFAULT 0.0,
    total_reviews INTEGER DEFAULT 0,
    profile_image VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Services offered by professionals
CREATE TABLE services (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price FLOAT,
    duration_minutes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Work experience
CREATE TABLE experiences (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) ON DELETE CASCADE,
    company VARCHAR(200),
    position VARCHAR(200),
    description TEXT,
    start_date DATE,
    end_date DATE,
    is_current BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Certifications
CREATE TABLE certifications (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    issuer VARCHAR(200),
    issue_date DATE,
    expiry_date DATE,
    credential_id VARCHAR(100),
    credential_url VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Comments/Reviews
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) NOT NULL,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    content TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Credits system
CREATE TABLE credits (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) NOT NULL,
    amount INTEGER DEFAULT 0,
    used INTEGER DEFAULT 0,
    transaction_type VARCHAR(50), -- 'purchase', 'usage', 'referral_bonus'
    transaction_amount INTEGER,
    payment_method VARCHAR(50), -- 'clabe', 'oxxo', 'efectivo'
    payment_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'confirmed', 'failed'
    price_mxn FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Referrals
CREATE TABLE referrals (
    id SERIAL PRIMARY KEY,
    referrer_id INTEGER REFERENCES professionals(id) NOT NULL,
    referred_user_id INTEGER REFERENCES users(id),
    referral_code VARCHAR(50) UNIQUE NOT NULL,
    commission_rate FLOAT DEFAULT 0.20,
    total_earned_mxn FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'expired', 'withdrawn'
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Referral earnings tracking
CREATE TABLE referral_earnings (
    id SERIAL PRIMARY KEY,
    referral_id INTEGER REFERENCES referrals(id) NOT NULL,
    credit_transaction_id INTEGER REFERENCES credits(id) NOT NULL,
    amount_mxn FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Referral withdrawals
CREATE TABLE referral_withdrawals (
    id SERIAL PRIMARY KEY,
    referrer_id INTEGER REFERENCES professionals(id) NOT NULL,
    amount_mxn FLOAT NOT NULL,
    withdrawal_method VARCHAR(50), -- 'clabe', 'oxxo', 'credits'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'completed'
    clabe_account VARCHAR(18),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Chatbot configuration
CREATE TABLE chatbot_configs (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    welcome_message TEXT,
    system_prompt TEXT,
    knowledge_base TEXT,
    max_tokens INTEGER DEFAULT 1000,
    temperature FLOAT DEFAULT 0.7,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages history
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    professional_id INTEGER REFERENCES professionals(id) NOT NULL,
    user_id INTEGER REFERENCES users(id),
    session_id VARCHAR(100),
    role VARCHAR(20), -- 'user', 'assistant'
    content TEXT NOT NULL,
    credits_used INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Categories for inquiries (existing)
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT
);

-- Inquiries (existing)
CREATE TABLE inquiries (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    category_id INTEGER REFERENCES categories(id),
    subject VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Responses (existing)
CREATE TABLE responses (
    id SERIAL PRIMARY KEY,
    inquiry_id INTEGER REFERENCES inquiries(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_professionals_user_id ON professionals(user_id);
CREATE INDEX idx_professionals_specialty ON professionals(specialty);
CREATE INDEX idx_professionals_city ON professionals(city);
CREATE INDEX idx_professionals_rating ON professionals(rating);
CREATE INDEX idx_comments_professional_id ON comments(professional_id);
CREATE INDEX idx_comments_status ON comments(status);
CREATE INDEX idx_credits_professional_id ON credits(professional_id);
CREATE INDEX idx_credits_payment_status ON credits(payment_status);
CREATE INDEX idx_referrals_code ON referrals(referral_code);
CREATE INDEX idx_chat_messages_professional_id ON chat_messages(professional_id);
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_inquiries_user_id ON inquiries(user_id);
CREATE INDEX idx_inquiries_status ON inquiries(status);
CREATE INDEX idx_responses_inquiry_id ON responses(inquiry_id);
