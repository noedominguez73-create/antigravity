from datetime import datetime
from app import db

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100))
    role = db.Column(db.String(20), default='user')  # 'user', 'professional', 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # OAuth fields
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    facebook_id = db.Column(db.String(255), unique=True, nullable=True)
    
    # Relationships
    inquiries = db.relationship('Inquiry', backref='user', lazy=True)
    responses = db.relationship('Response', backref='responder', lazy=True)
    professional_profile = db.relationship('Professional', backref='user', uselist=False)
    comments = db.relationship('Comment', backref='author', lazy=True)
    store_profile = db.relationship('Store', backref='user', uselist=False)

class Store(db.Model):
    __tablename__ = 'stores'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    store_type = db.Column(db.String(100)) # Can be linked to StoreType or just string
    store_name = db.Column(db.String(200)) # Optional if different from user.full_name
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_stores_user_id', 'user_id'),
        db.Index('idx_stores_store_type', 'store_type'),
    )

class StoreType(db.Model):
    __tablename__ = 'store_types'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Professional(db.Model):
    __tablename__ = 'professionals'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    specialty = db.Column(db.String(100))
    city = db.Column(db.String(100))
    bio = db.Column(db.Text)
    rating = db.Column(db.Float, default=0.0)
    total_reviews = db.Column(db.Integer, default=0)
    balance = db.Column(db.Integer, default=0) # Ledger balance for atomic transactions
    profile_image = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    services = db.relationship('Service', backref='professional', lazy=True, cascade='all, delete-orphan')
    experiences = db.relationship('Experience', backref='professional', lazy=True, cascade='all, delete-orphan')
    certifications = db.relationship('Certification', backref='professional', lazy=True, cascade='all, delete-orphan')
    comments = db.relationship('Comment', backref='professional', lazy=True)
    credits = db.relationship('Credit', backref='professional', lazy=True)
    referrals_made = db.relationship('Referral', foreign_keys='Referral.referrer_id', backref='referrer', lazy=True)
    chatbot_config = db.relationship('ChatbotConfig', backref='professional', uselist=False)
    
    __table_args__ = (
        db.Index('idx_professionals_user_id', 'user_id'),
        db.Index('idx_professionals_city', 'city'),
        db.Index('idx_professionals_specialty', 'specialty'),
    )

class Service(db.Model):
    __tablename__ = 'services'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float)
    duration_minutes = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_services_professional_id', 'professional_id'),
    )

class Experience(db.Model):
    __tablename__ = 'experiences'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    company = db.Column(db.String(200))
    position = db.Column(db.String(200))
    description = db.Column(db.Text)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date, nullable=True)
    is_current = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_experiences_professional_id', 'professional_id'),
    )

class Certification(db.Model):
    __tablename__ = 'certifications'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    issuer = db.Column(db.String(200))
    issue_date = db.Column(db.Date)
    expiry_date = db.Column(db.Date, nullable=True)
    credential_id = db.Column(db.String(100))
    credential_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_certifications_professional_id', 'professional_id'),
    )

class Comment(db.Model):
    __tablename__ = 'comments'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # 1-5
    content = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'approved', 'rejected'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_comments_professional_id', 'professional_id'),
        db.Index('idx_comments_user_id', 'user_id'),
        db.Index('idx_comments_status', 'status'),
    )

class Credit(db.Model):
    __tablename__ = 'credits'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    amount = db.Column(db.Integer, default=0)  # Total credits
    used = db.Column(db.Integer, default=0)  # Used credits
    transaction_type = db.Column(db.String(50))  # 'purchase', 'usage', 'referral_bonus'
    transaction_amount = db.Column(db.Integer)  # Credits in this transaction
    payment_method = db.Column(db.String(50))  # 'clabe', 'oxxo', 'efectivo'
    payment_status = db.Column(db.String(20), default='pending')  # 'pending', 'confirmed', 'failed'
    price_mxn = db.Column(db.Float)  # Amount paid in MXN
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_credits_professional_id', 'professional_id'),
        db.Index('idx_credits_payment_status', 'payment_status'),
    )

class Referral(db.Model):
    __tablename__ = 'referrals'
    
    id = db.Column(db.Integer, primary_key=True)
    referrer_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    referred_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    referral_code = db.Column(db.String(50), unique=True, nullable=False)
    commission_rate = db.Column(db.Float, default=0.20)  # 20%
    total_earned_mxn = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='active')  # 'active', 'expired', 'withdrawn'
    expires_at = db.Column(db.DateTime)  # 12 months from creation
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_referrals_referrer_id', 'referrer_id'),
        db.Index('idx_referrals_referred_user_id', 'referred_user_id'),
        db.Index('idx_referrals_code', 'referral_code'),
    )

class ReferralEarning(db.Model):
    __tablename__ = 'referral_earnings'
    
    id = db.Column(db.Integer, primary_key=True)
    referral_id = db.Column(db.Integer, db.ForeignKey('referrals.id'), nullable=False)
    credit_transaction_id = db.Column(db.Integer, db.ForeignKey('credits.id'), nullable=False)
    amount_mxn = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ReferralWithdrawal(db.Model):
    __tablename__ = 'referral_withdrawals'
    
    id = db.Column(db.Integer, primary_key=True)
    referrer_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    amount_mxn = db.Column(db.Float, nullable=False)
    withdrawal_method = db.Column(db.String(50))  # 'clabe', 'oxxo', 'credits'
    status = db.Column(db.String(20), default='pending')  # 'pending', 'approved', 'completed'
    clabe_account = db.Column(db.String(18), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime, nullable=True)

class ChatbotConfig(db.Model):
    __tablename__ = 'chatbot_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    welcome_message = db.Column(db.Text)
    system_prompt = db.Column(db.Text)
    knowledge_base = db.Column(db.Text)  # Stored documents/context
    max_tokens = db.Column(db.Integer, default=1000)
    temperature = db.Column(db.Float, default=0.7)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    session_id = db.Column(db.String(100))
    role = db.Column(db.String(20))  # 'user', 'assistant'
    content = db.Column(db.Text, nullable=False)
    credits_used = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_chat_messages_professional_id', 'professional_id'),
        db.Index('idx_chat_messages_session_id', 'session_id'),
    )

# Keep existing models
class Category(db.Model):
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    
    # Relationships
    inquiries = db.relationship('Inquiry', backref='category', lazy=True)

class Inquiry(db.Model):
    __tablename__ = 'inquiries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'))
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    responses = db.relationship('Response', backref='inquiry', lazy=True, cascade='all, delete-orphan')

class Response(db.Model):
    __tablename__ = 'responses'
    
    id = db.Column(db.Integer, primary_key=True)
    inquiry_id = db.Column(db.Integer, db.ForeignKey('inquiries.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)



class KnowledgeBaseDocument(db.Model):
    __tablename__ = 'knowledge_base_documents'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    text_content = db.Column(db.Text, nullable=True)
    file_size = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Indices for faster lookup
    __table_args__ = (
        db.Index('idx_kbdoc_professional_id', 'professional_id'),
        db.Index('idx_kbdoc_uploaded_at', 'uploaded_at'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.original_filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'size_formatted': self._format_size(self.file_size),
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None
        }
    
    def _format_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

class ProfessionalURL(db.Model):
    __tablename__ = 'professional_urls'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    specialty = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    last_fetched = db.Column(db.DateTime, nullable=True)
    cached_content = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_prof_urls_professional_id', 'professional_id'),
        db.Index('idx_prof_urls_is_active', 'is_active'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'specialty': self.specialty,
            'description': self.description,
            'is_active': self.is_active,
            'last_fetched': self.last_fetched.isoformat() if self.last_fetched else None,
            'created_at': self.created_at.isoformat()
        }




class CachedResponse(db.Model):
    __tablename__ = 'cached_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    professional_id = db.Column(db.Integer, db.ForeignKey('professionals.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.PickleType, nullable=True) # Store numpy array if needed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_cached_responses_prof_id', 'professional_id'),
    )

class KnowledgeBaseChunk(db.Model):
    __tablename__ = 'knowledge_base_chunks'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('knowledge_base_documents.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.PickleType, nullable=False) # Store numpy array
    chunk_index = db.Column(db.Integer, nullable=False)
    
    __table_args__ = (
        db.Index('idx_kb_chunks_doc_id', 'document_id'),
    )

class Specialty(db.Model):
    __tablename__ = 'specialties'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MirrorItem(db.Model):
    __tablename__ = 'mirror_items'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False) # 'hairstyle', 'color'
    image_url = db.Column(db.String(500), nullable=True) # Optional URL or path
    color_code = db.Column(db.String(20), nullable=True) # Hex code for colors
    prompt = db.Column(db.Text, nullable=True) # Generated prompt from Gemini
    order_index = db.Column(db.Integer, default=0) # For custom sorting
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_mirror_items_category', 'category'),
        db.Index('idx_mirror_items_is_active', 'is_active'),
        db.Index('idx_mirror_items_order', 'order_index'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'image_url': self.image_url,
            'color_code': self.color_code,
            'prompt': self.prompt,
            'order_index': self.order_index,
            'is_active': self.is_active
        }

class MirrorUsage(db.Model):
    __tablename__ = 'mirror_usages'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    usage_type = db.Column(db.String(50), default='generation') # 'generation', 'view'
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True) # Anonymous usage allowed?
    item_id = db.Column(db.Integer, db.ForeignKey('mirror_items.id'), nullable=True) # Specific item used
    
    __table_args__ = (
        db.Index('idx_mirror_usage_created_at', 'created_at'),
    )
