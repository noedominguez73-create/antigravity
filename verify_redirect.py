
from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) # Run headless for speed, or false for debugging if needed
        context = browser.new_context()
        page = context.new_page()

        # 1. Visit Homepage - not logged in
        print("1. Visiting Homepage (not logged in)...")
        page.goto("http://localhost:5000")
        
        # Verify Auth Buttons visible, User Menu hidden
        auth_buttons = page.locator("#authButtons")
        user_menu = page.locator("#userMenu")
        
        if auth_buttons.is_visible() and not user_menu.is_visible():
            print("SUCCESS: Auth buttons visible, User Menu hidden.")
        else:
            print(f"FAILURE: Initial state incorrect. Auth: {auth_buttons.is_visible()}, Menu: {user_menu.is_visible()}")

        # 2. Go to Login
        print("2. Navigating to Login...")
        page.goto("http://localhost:5000/login")
        
        # 3. Perform Login
        # Note: Assuming 'test@example.com' / 'Password123!' exists or creating it if needed. 
        # For this test to be robust, we probably should register a new user or rely on a known one.
        # Let's try to register a fresh user to be safe.
        
        print("3. Navigating to Register...")
        page.goto("http://localhost:5000/registro")
        
        timestamp = int(time.time())
        email = f"testuser_{timestamp}@example.com"
        password = "Password123!"
        
        print(f"   Registering user: {email}")
        page.fill("#fullName", "Test User Auto")
        page.fill("#email", email)
        page.fill("#password", password)
        page.fill("#confirmPassword", password)
        page.check("#terms")
        
        # Select 'Usuario' role (default, but confirming)
        # page.click("#userRoleBtn") 

        page.click("#submitBtn")
        
        # 4. Verify Redirect
        print("4. Verifying redirection...")
        # Wait for navigation or specific URL pattern
        try:
            page.wait_for_url("**/perfil-usuario", timeout=5000)
            print("SUCCESS: Redirected to /perfil-usuario")
        except:
            print(f"FAILURE: Did not redirect to /perfil-usuario. Current URL: {page.url}")
            # If failed, capture screenshot or page content?
            # page.screenshot(path="redirect_fail.png")

        # 5. Visit Homepage - logged in
        print("5. Visiting Homepage (logged in)...")
        page.goto("http://localhost:5000")
        
        # Reload locators
        auth_buttons = page.locator("#authButtons")
        user_menu = page.locator("#userMenu")
        
        # User Menu should now be visible (because of auth.js script running on index.html)
        # We need to wait a moment for the JS to execute updateAuthUI
        try:
             user_menu.wait_for(state="visible", timeout=2000)
             print("SUCCESS: User Menu is visible.")
        except:
             print("FAILURE: User Menu not visible after login.")

        if not auth_buttons.is_visible():
             print("SUCCESS: Auth buttons hidden.")
        else:
             print("FAILURE: Auth buttons still visible.")

        browser.close()

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}")
