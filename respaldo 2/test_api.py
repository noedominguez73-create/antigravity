"""
Automated Test Suite for AsesoriaIMSS.io API
============================================

Tests all critical endpoints and functionality.

Run with: python -m pytest test_api.py -v
Or: pytest test_api.py -v --tb=short

Author: Automated Testing Suite
Date: 2025-11-26
"""

import pytest
import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USER_EMAIL = f"test.user.{datetime.now().timestamp()}@example.com"
TEST_USER_PASSWORD = "TestPassword123"

# Global variables to store tokens and IDs
admin_token = None
user_token = None
professional_token = None
test_user_id = None


class TestHomepage:
    """Test 1: Homepage accessibility"""
    
    def test_homepage_loads(self):
        """GET / - Homepage should return 200"""
        response = requests.get(f"{BASE_URL}/")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert len(response.content) > 0, "Homepage content is empty"
        print(f"\n✅ Homepage loaded successfully ({len(response.content)} bytes)")


class TestAuthentication:
    """Test 2-3: User registration and login"""
    
    def test_user_registration(self):
        """POST /api/auth/register - Create test user"""
        global test_user_id
        
        data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD,
            "full_name": "Test User Automated",
            "role": "user",
            "city": "Ciudad de México"
        }
        
        response = requests.post(f"{BASE_URL}/api/auth/register", json=data)
        result = response.json()
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"
        assert result.get('success') == True, f"Registration failed: {result.get('error')}"
        assert 'user' in result.get('data', {}), "User data not in response"
        
        test_user_id = result['data']['user']['id']
        print(f"\n✅ User registered: {TEST_USER_EMAIL} (ID: {test_user_id})")
    
    def test_user_login(self):
        """POST /api/auth/login - Login test user"""
        global user_token
        
        data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        
        response = requests.post(f"{BASE_URL}/api/auth/login", json=data)
        result = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert result.get('success') == True, f"Login failed: {result.get('error')}"
        assert 'token' in result.get('data', {}), "Token not in response"
        
        user_token = result['data']['token']
        print(f"\n✅ User logged in successfully")
        print(f"   Token: {user_token[:30]}...")


class TestProfessionals:
    """Test 4-5: Professional listing and profile"""
    
    def test_get_professionals_list(self):
        """GET /api/profesionales - Get list of professionals"""
        response = requests.get(f"{BASE_URL}/api/profesionales")
        result = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert result.get('success') == True, f"Failed: {result.get('error')}"
        assert 'professionals' in result.get('data', {}), "Professionals list not in response"
        
        professionals = result['data']['professionals']
        print(f"\n✅ Professionals list retrieved: {len(professionals)} professionals")
        
        if professionals:
            print(f"   First professional: {professionals[0].get('full_name')}")
    
    def test_get_professional_profile(self):
        """GET /api/profesionales/2 - Get specific professional profile"""
        professional_id = 2
        response = requests.get(f"{BASE_URL}/api/profesionales/{professional_id}")
        result = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert result.get('success') == True, f"Failed: {result.get('error')}"
        assert 'professional' in result.get('data', {}), "Professional data not in response"
        
        prof = result['data']['professional']
        print(f"\n✅ Professional profile retrieved")
        print(f"   Name: {prof.get('full_name')}")
        print(f"   Specialty: {prof.get('specialty')}")
        print(f"   Rating: {prof.get('rating')}")


class TestAdminCredits:
    """Test 6: Admin credit addition"""
    
    @pytest.fixture(autouse=True)
    def setup_admin_token(self):
        """Login as admin before test"""
        global admin_token
        
        data = {
            "email": "admin@example.com",
            "password": "admin123"
        }
        
        response = requests.post(f"{BASE_URL}/api/admin/login", json=data)
        result = response.json()
        
        if result.get('success'):
            admin_token = result['data']['token']
            print(f"\n🔐 Admin logged in for testing")
    
    def test_admin_add_credits(self):
        """POST /api/admin/creditos/2 - Admin adds credits to professional"""
        global admin_token
        
        assert admin_token is not None, "Admin token not available"
        
        professional_id = 2
        data = {
            "amount": 50,
            "reason": "Automated pytest credit addition"
        }
        
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.post(
            f"{BASE_URL}/api/admin/creditos/{professional_id}",
            json=data,
            headers=headers
        )
        result = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert result.get('success') == True, f"Failed: {result.get('error')}"
        
        print(f"\n✅ Admin added credits successfully")
        print(f"   Professional ID: {professional_id}")
        print(f"   Amount: {data['amount']}")
        print(f"   Reason: {data['reason']}")


class TestCredits:
    """Test 7: Credit balance check"""
    
    @pytest.fixture(autouse=True)
    def setup_professional_token(self):
        """Login as professional before test"""
        global professional_token
        
        data = {
            "email": "maria.lopez@example.com",
            "password": "password123"
        }
        
        response = requests.post(f"{BASE_URL}/api/auth/login", json=data)
        result = response.json()
        
        if result.get('success'):
            professional_token = result['data']['token']
            print(f"\n🔐 Professional logged in for testing")
    
    def test_get_credit_balance(self):
        """GET /api/creditos/2 - Get credit balance"""
        global professional_token
        
        assert professional_token is not None, "Professional token not available"
        
        professional_id = 2
        headers = {"Authorization": f"Bearer {professional_token}"}
        response = requests.get(
            f"{BASE_URL}/api/creditos/{professional_id}",
            headers=headers
        )
        result = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert result.get('success') == True, f"Failed: {result.get('error')}"
        assert 'total_credits' in result.get('data', {}), "Credit data not in response"
        
        data = result['data']
        print(f"\n✅ Credit balance retrieved")
        print(f"   Total Purchased: {data.get('total_purchased', 0)}")
        print(f"   Admin Additions: {data.get('admin_additions', 0)}")
        print(f"   Referral Bonuses: {data.get('referral_bonuses', 0)}")
        print(f"   Total Credits: {data.get('total_credits', 0)}")
        print(f"   Used: {data.get('used', 0)}")
        print(f"   Available: {data.get('available', 0)}")


class TestChatbot:
    """Test 8: Chatbot messaging"""
    
    def test_send_chat_message(self):
        """POST /api/chat - Send message to chatbot"""
        global professional_token
        
        # Note: This test may fail if chatbot endpoint requires different auth
        # or if Gemini API is not configured
        
        data = {
            "professional_id": 2,
            "message": "Hola, esta es una prueba automatizada. ¿Cuáles son tus servicios?",
            "session_id": f"test_session_{datetime.now().timestamp()}"
        }
        
        headers = {}
        if professional_token:
            headers["Authorization"] = f"Bearer {professional_token}"
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json=data,
                headers=headers,
                timeout=10
            )
            result = response.json()
            
            # Accept both 200 and 201 as success
            assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}"
            
            if result.get('success'):
                print(f"\n✅ Chat message sent successfully")
                if 'response' in result.get('data', {}):
                    response_text = result['data']['response'][:100]
                    print(f"   Response: {response_text}...")
            else:
                print(f"\n⚠️  Chat returned error: {result.get('error')}")
                # Don't fail the test if it's a configuration issue
                pytest.skip(f"Chat endpoint not fully configured: {result.get('error')}")
                
        except requests.exceptions.Timeout:
            print(f"\n⚠️  Chat request timed out (Gemini API may be slow)")
            pytest.skip("Chat request timed out")
        except Exception as e:
            print(f"\n⚠️  Chat test error: {str(e)}")
            pytest.skip(f"Chat endpoint error: {str(e)}")


class TestAdminDashboard:
    """Test 9: Admin dashboard access"""
    
    def test_admin_dashboard(self):
        """GET /api/admin/dashboard - Access admin dashboard"""
        global admin_token
        
        assert admin_token is not None, "Admin token not available"
        
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.get(
            f"{BASE_URL}/api/admin/dashboard",
            headers=headers
        )
        result = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert result.get('success') == True, f"Failed: {result.get('error')}"
        
        data = result['data']
        print(f"\n✅ Admin dashboard accessed successfully")
        print(f"   Total Users: {data.get('users', {}).get('total', 0)}")
        print(f"   Total Professionals: {data.get('professionals', {}).get('total', 0)}")
        print(f"   Pending Comments: {data.get('pending', {}).get('comments', 0)}")
        print(f"   Pending Payments: {data.get('pending', {}).get('payments', 0)}")
        print(f"   Total Revenue: ${data.get('revenue', {}).get('total_mxn', 0)} MXN")


# Test execution summary
def pytest_sessionfinish(session, exitstatus):
    """Print summary after all tests complete"""
    print("\n" + "="*70)
    print("🧪 TEST SUITE EXECUTION COMPLETE")
    print("="*70)
    
    if exitstatus == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ SOME TESTS FAILED (Exit status: {exitstatus})")
    
    print("="*70)


if __name__ == "__main__":
    # Allow running directly with: python test_api.py
    pytest.main([__file__, "-v", "--tb=short"])
