# 🧪 Testing Execution Guide - AsesoriaIMSS.io

## Quick Start Testing Checklist

### ✅ Pre-Testing Setup
- [x] Server running on http://localhost:5000
- [ ] Database initialized with seed data
- [ ] Gemini API key configured in .env
- [ ] Browser ready (Chrome/Firefox recommended)

---

## 🎯 Priority Tests (Start Here)

### Test 1: Admin Login & Dashboard
**URL:** http://localhost:5000/admin/login

**Credentials:**
```
Email: admin@example.com
Password: admin123
```

**Expected Results:**
- ✅ Login form renders (not raw HTML)
- ✅ Successful login redirects to /admin
- ✅ Dashboard shows statistics
- ✅ Three tabs visible: Comentarios, Agregar Créditos, Transacciones

**Status:** [ ] PASS / [ ] FAIL

---

### Test 2: Admin Add Credits
**Prerequisites:** Logged in as admin

**Steps:**
1. Click tab "Agregar Créditos"
2. Fill form:
   - ID del Profesional: `2`
   - Cantidad: `100`
   - Razón: `Testing credits addition`
3. Click "Agregar Créditos"

**Expected Results:**
- ✅ Success message: "100 créditos agregados exitosamente"
- ✅ No errors in console
- ✅ Transaction recorded

**Status:** [ ] PASS / [ ] FAIL

---

### Test 3: Verify Credits in Professional Account
**URL:** http://localhost:5000/login

**Credentials:**
```
Email: maria.lopez@example.com
Password: password123
```

**Steps:**
1. Login as professional
2. Navigate to: http://localhost:5000/creditos

**Expected Results:**
- ✅ Credit balance increased by 100
- ✅ Transaction history shows "admin_addition"
- ✅ Status shows "confirmed" (green)
- ✅ Description: "Testing credits addition"

**Status:** [ ] PASS / [ ] FAIL

---

### Test 4: Professional Profile & Chatbot
**URL:** http://localhost:5000/profesional?id=2

**Steps:**
1. Visit professional profile (logged in or not)
2. Look for chatbot widget in bottom-right corner
3. Click to open chat
4. Send message: "Hola, ¿cuáles son tus servicios?"

**Expected Results:**
- ✅ Profile loads correctly
- ✅ Chatbot widget visible
- ✅ Chat opens when clicked
- ✅ Gemini responds within 5 seconds
- ✅ Response is relevant

**Status:** [ ] PASS / [ ] FAIL

---

### Test 5: Credit Consumption
**Prerequisites:** Professional logged in with credits

**Steps:**
1. As professional, check current balance at /creditos
2. Note the number (e.g., 150 credits)
3. Open chatbot on own profile
4. Send 3 messages
5. Refresh /creditos page

**Expected Results:**
- ✅ Balance decreased by 3 credits
- ✅ Transaction history shows chat usage
- ✅ Each message = 1 credit

**Status:** [ ] PASS / [ ] FAIL

---

## 🔍 Secondary Tests

### Test 6: User Registration
**URL:** http://localhost:5000/registro

**Test Data:**
```
Tipo: Usuario/Cliente
Nombre: Test User
Email: testuser@example.com
Password: Test123456
Ciudad: Ciudad de México
```

**Status:** [ ] PASS / [ ] FAIL

---

### Test 7: Professional Search
**URL:** http://localhost:5000

**Steps:**
1. Use search form
2. Select specialty and city
3. Click "Buscar Profesionales"

**Expected Results:**
- ✅ Results page loads
- ✅ Professionals displayed
- ✅ Each has name, specialty, rating

**Status:** [ ] PASS / [ ] FAIL

---

### Test 8: Admin Transactions Tab
**Prerequisites:** Admin logged in

**Steps:**
1. Go to /admin
2. Click "Transacciones Pendientes" tab
3. Check if any pending transactions appear

**Expected Results:**
- ✅ Tab loads without errors
- ✅ Shows pending transactions OR "No hay transacciones pendientes"
- ✅ If transactions exist, Aprobar/Rechazar buttons work

**Status:** [ ] PASS / [ ] FAIL

---

## 🐛 Bug Tracking

### Known Fixed Issues
1. ✅ Admin /admin route - Fixed HTML rendering
2. ✅ Admin /admin/login route - Created route
3. ✅ Admin credit addition - Fixed payment_status from 'completed' to 'confirmed'
4. ✅ Credit calculation - Added admin_additions to total

### Issues to Watch For
- [ ] Chatbot widget not appearing
- [ ] Gemini API errors
- [ ] Credit balance not updating
- [ ] Transaction history incorrect
- [ ] Login redirects not working

---

## 📝 Testing Notes Template

```
Test Date: ___________
Tester: ___________
Browser: ___________

Test #: _____
Status: PASS / FAIL
Notes: 
_______________________________
_______________________________

Screenshots: (if applicable)
```

---

## 🚀 Quick Test Commands

### Check if server is running:
```bash
curl http://localhost:5000
```

### Check admin login endpoint:
```bash
curl http://localhost:5000/admin/login
```

### Test admin API login:
```bash
curl -X POST http://localhost:5000/api/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"admin123"}'
```

---

## ✅ Final Validation Checklist

### Critical Functionality
- [ ] Admin can login
- [ ] Admin can add credits
- [ ] Credits appear in professional account
- [ ] Chatbot responds
- [ ] Credits are consumed

### User Experience
- [ ] All pages render properly (no raw HTML)
- [ ] Forms submit successfully
- [ ] Error messages are clear
- [ ] Success notifications appear

### Performance
- [ ] Pages load < 2 seconds
- [ ] Chatbot responds < 5 seconds
- [ ] No console errors
- [ ] No 500 errors

---

## 🎯 Test Execution Order

**Recommended sequence:**
1. Admin Login (Test 1)
2. Admin Add Credits (Test 2)
3. Verify Credits (Test 3)
4. Chatbot Test (Test 4)
5. Credit Consumption (Test 5)
6. Secondary tests as needed

**Estimated time:** 15-20 minutes for priority tests

---

## 📊 Results Summary

**Total Tests Executed:** _____
**Passed:** _____
**Failed:** _____
**Blocked:** _____

**Overall Status:** ⬜ PASS / ⬜ FAIL / ⬜ PARTIAL

---

**Last Updated:** 2025-11-26
**Server Status:** ✅ Running on http://localhost:5000
