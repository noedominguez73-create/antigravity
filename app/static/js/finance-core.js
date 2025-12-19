/**
 * FinanceCore - Central Logic for Mis Finanzas
 * Handles data persistence (localStorage) and business logic.
 * Production-Ready v2.0
 */

const FinanceCore = {
    // --- Configuration ---
    config: {
        debug: true, // Set to false in production
        storageKey: 'misFinanzasData'
    },

    data: {
        accounts: [],
        transactions: [],
        bills: [],
        receivables: [],
        todos: [],
        categories: [
            { id: '1', name: 'Supermercado', icon: 'shopping-cart', color: '#10b981', subcategories: [] },
            { id: '2', name: 'Transporte', icon: 'car', color: '#3b82f6', subcategories: ['Gasolina', 'Mantenimiento', 'Uber'] },
            { id: '3', name: 'Servicios', icon: 'zap', color: '#f59e0b', subcategories: ['Luz', 'Agua', 'Internet'] },
            { id: '4', name: 'Entretenimiento', icon: 'film', color: '#8b5cf6', subcategories: ['Cine', 'Streaming'] },
            { id: '5', name: 'Salud', icon: 'heart', color: '#ef4444', subcategories: ['Farmacia', 'Consultas'] },
            { id: '6', name: 'Educación', icon: 'book', color: '#6366f1', subcategories: [] },
            { id: '7', name: 'Ropa', icon: 'shirt', color: '#ec4899', subcategories: ['Zapatos', 'Accesorios'] },
            { id: '8', name: 'Hogar', icon: 'home', color: '#f97316', subcategories: ['Muebles', 'Limpieza'] },
            { id: '9', name: 'Otros', icon: 'box', color: '#6b7280', subcategories: [] }
        ]
    },

    // --- Utilities ---
    log(message, ...args) {
        if (this.config.debug) {
            console.log(`[FinanceCore] ${message}`, ...args);
        }
    },

    error(message, ...args) {
        console.error(`[FinanceCore Error] ${message}`, ...args);
        if (this.showToast) this.showToast(message, 'error');
    },

    sanitizeInput(input) {
        if (typeof input !== 'string') return input;
        return input.replace(/[<>]/g, '').trim(); // Prevent basic HTML injection
    },

    showToast: null, // UI can override this to show toasts

    // --- Persistence ---
    init() {
        try {
            this.loadData();
            if (this.data.accounts.length === 0) {
                this.seedData();
            }
            this.log("Initialized", this.data);
            this.notifyChange();

            window.addEventListener('storage', (e) => {
                if (e.key === this.config.storageKey) {
                    this.log("Syncing data from another tab...");
                    this.loadData();
                    this.notifyChange();
                }
            });
        } catch (e) {
            this.error("Initialization failed", e);
        }
    },

    loadData() {
        try {
            const stored = localStorage.getItem(this.config.storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Validation Step: Ensure structure integrity
                if (!parsed || typeof parsed !== 'object') throw new Error("Corrupted data");

                this.data = { ...this.data, ...parsed };

                // Restore Dates
                if (this.data.transactions) this.data.transactions.forEach(t => t.date = new Date(t.date));
                if (this.data.clients) this.data.clients.forEach(c => c.createdAt = new Date(c.createdAt));

                this.migrateData();
            }
        } catch (e) {
            this.error("Failed to load data", e);
        }
    },

    saveData() {
        try {
            localStorage.setItem(this.config.storageKey, JSON.stringify(this.data));
            this.notifyChange();
        } catch (e) {
            this.error("Failed to save data. Storage might be full.", e);
        }
    },

    migrateData() {
        let changed = false;
        // Migration: Ensure all categories have IDs
        if (this.data.categories) {
            this.data.categories.forEach((cat, index) => {
                if (!cat.id) { cat.id = `cat-${Date.now()}-${index}`; changed = true; }
                if (!cat.subcategories) { cat.subcategories = []; changed = true; }
            });
        }
        if (changed) this.saveData();
    },

    seedData() {
        this.data.accounts = [
            { id: '1', name: 'Efectivo', type: 'cash', balance: 500, color: '#10b981' },
            { id: '2', name: 'Banco', type: 'debit', balance: 12000, color: '#3b82f6' }
        ];
        this.saveData();
    },

    // --- Validation Helpers ---
    validateBill(bill) {
        if (!bill.name || bill.name.length < 2) throw new Error("El nombre de la factura es obligatorio.");
        if (isNaN(bill.amount) || bill.amount <= 0) throw new Error("El monto debe ser mayor a 0.");
        if (!bill.dueDate) throw new Error("La fecha de vencimiento es obligatoria.");

        // Duplicate Check (Name + Amount + DueDate)
        const duplicate = this.data.bills.find(b =>
            b.name.toLowerCase() === bill.name.toLowerCase() &&
            Math.abs(b.amount - bill.amount) < 0.01 &&
            b.dueDate === bill.dueDate &&
            b.id !== bill.id // Exclude self if updating
        );
        if (duplicate) throw new Error("Ya existe una factura similar (Nombre, Monto y Fecha idénticos).");

        return true;
    },

    // --- Category Management ---
    addCategory(categoryData) {
        const id = Date.now().toString();
        const newCategory = {
            id,
            name: this.sanitizeInput(categoryData.name),
            icon: categoryData.icon || 'circle',
            color: categoryData.color || '#6b7280',
            subcategories: categoryData.subcategories || []
        };
        this.data.categories.push(newCategory);
        this.saveData();
        return newCategory;
    },

    updateCategory(id, updates) {
        const index = this.data.categories.findIndex(c => c.id === id);
        if (index !== -1) {
            if (updates.name) updates.name = this.sanitizeInput(updates.name);
            this.data.categories[index] = { ...this.data.categories[index], ...updates };
            this.saveData();
            return true;
        }
        return false;
    },

    deleteCategory(id) {
        const index = this.data.categories.findIndex(c => c.id === id);
        if (index !== -1) {
            this.data.categories.splice(index, 1);
            this.saveData();
            return true;
        }
        return false;
    },

    // --- Accounts ---
    addAccount(name, type, balance) {
        const id = Date.now().toString();
        this.data.accounts.push({
            id,
            name: this.sanitizeInput(name),
            type,
            balance: parseFloat(balance),
            color: '#6b7280'
        });
        this.saveData();
        return id;
    },

    updateAccount(id, updates) {
        const index = this.data.accounts.findIndex(a => a.id === id);
        if (index !== -1) {
            if (updates.name) updates.name = this.sanitizeInput(updates.name);
            this.data.accounts[index] = { ...this.data.accounts[index], ...updates };
            this.saveData();
            return true;
        }
        return false;
    },

    getAccounts() {
        return this.data.accounts;
    },

    getTotalBalance() {
        return this.data.accounts.reduce((sum, acc) => sum + acc.balance, 0);
    },

    transfer(fromAccountId, toAccountId, amount) {
        const fromAccount = this.data.accounts.find(a => a.id === fromAccountId);
        const toAccount = this.data.accounts.find(a => a.id === toAccountId);
        const numAmount = parseFloat(amount);

        if (fromAccount && toAccount && numAmount > 0 && fromAccount.balance >= numAmount) {
            // 1. Expense from Source
            this.addTransaction('expense', numAmount, 'Transferencia', `Transferencia a ${toAccount.name}`, fromAccountId);

            // 2. Income to Destination
            this.addTransaction('income', numAmount, 'Transferencia', `Transferencia de ${fromAccount.name}`, toAccountId);

            return true;
        }
        return false;
    },

    // --- Transactions ---
    addTransaction(type, amount, category, description, accountId, date = null) {
        try {
            const transaction = {
                id: Date.now().toString(),
                date: date ? new Date(date) : new Date(),
                type, // 'income' or 'expense'
                amount: parseFloat(amount),
                category,
                description: this.sanitizeInput(description),
                accountId
            };

            if (transaction.amount <= 0) throw new Error("El monto debe ser positivo.");

            // Update Account Balance
            const account = this.data.accounts.find(a => a.id === accountId);
            if (account) {
                if (type === 'income') {
                    account.balance += transaction.amount;
                } else {
                    account.balance -= transaction.amount;
                }
            }

            this.data.transactions.unshift(transaction);
            this.saveData();
            return transaction;
        } catch (e) {
            this.error("Error adding transaction", e);
            throw e;
        }
    },

    getTransactions() {
        return this.data.transactions;
    },

    getRecentTransactions(limit = 5) {
        return this.data.transactions.slice(0, limit);
    },

    deleteTransaction(id) {
        const index = this.data.transactions.findIndex(t => t.id === id);
        if (index !== -1) {
            const t = this.data.transactions[index];
            // Reverse balance impact
            const account = this.data.accounts.find(a => a.id === t.accountId);
            if (account) {
                if (t.type === 'income') account.balance -= t.amount;
                else account.balance += t.amount;
            }
            this.data.transactions.splice(index, 1);
            this.saveData();
        }
    },

    // --- Bills ---
    addBill(billData) {
        try {
            let bill = {};
            const id = Date.now().toString();

            // Handle legacy or object args
            if (arguments.length > 1 && typeof arguments[0] === 'string') {
                bill = {
                    id,
                    name: arguments[0],
                    amount: parseFloat(arguments[1]),
                    dueDate: arguments[2],
                    isPaid: false,
                    category: 'Servicios',
                    recurring: false,
                    notes: '',
                    paidAmount: 0,
                    payments: []
                };
            } else {
                bill = {
                    id,
                    name: billData.name,
                    amount: parseFloat(billData.amount || 0),
                    dueDate: billData.dueDate,
                    category: billData.category || 'Servicios',
                    subcategory: billData.subcategory || '',
                    recurring: billData.recurring || false,
                    frequency: billData.frequency || 'monthly',
                    previousAmount: parseFloat(billData.previousAmount || 0),
                    notes: billData.notes || '',
                    isPaid: false,
                    paidAmount: 0,
                    payments: []
                };
            }

            bill.name = this.sanitizeInput(bill.name);
            bill.notes = this.sanitizeInput(bill.notes);

            this.validateBill(bill); // Will throw if invalid

            this.data.bills.push(bill);
            this.saveData();
            this.log("Bill added", bill);
            return bill;
        } catch (e) {
            this.error(e.message);
            throw e; // Re-throw so UI can handle it
        }
    },

    updateBill(billData) {
        try {
            const index = this.data.bills.findIndex(b => b.id === billData.id);
            if (index === -1) throw new Error("Factura no encontrada.");

            const existing = this.data.bills[index];
            const updated = {
                ...existing,
                ...billData,
                name: this.sanitizeInput(billData.name || existing.name),
                notes: this.sanitizeInput(billData.notes || existing.notes),
                amount: parseFloat(billData.amount !== undefined ? billData.amount : existing.amount)
            };

            this.validateBill(updated);

            // Re-check isPaid status
            updated.isPaid = updated.paidAmount >= updated.amount - 0.01;

            this.data.bills[index] = updated;
            this.saveData();
            return true;
        } catch (e) {
            this.error(e.message);
            throw e;
        }
    },

    attachReceipt(billId, imageData) {
        const bill = this.data.bills.find(b => b.id === billId);
        if (bill) {
            bill.receipt = imageData;
            this.saveData();
        }
    },

    attachTransactionDocument(transactionId, imageData) {
        const transaction = this.data.transactions.find(t => t.id === transactionId);
        if (transaction) {
            transaction.document = imageData;
            this.saveData();
        }
    },

    payBill(billId, accountId, amount) {
        const bill = this.data.bills.find(b => b.id === billId);
        const account = this.data.accounts.find(a => a.id === accountId);
        const payAmount = parseFloat(amount);

        if (bill && account && payAmount > 0) {
            // Create Expense
            const transaction = this.addTransaction('expense', payAmount, bill.category || 'Servicios', `Pago parcial: ${bill.name}`, accountId);

            // Update Bill
            if (!bill.payments) bill.payments = [];
            if (!bill.paidAmount) bill.paidAmount = 0;

            bill.payments.push({
                id: Date.now().toString(),
                transactionId: transaction.id,
                date: new Date(),
                amount: payAmount,
                accountId: accountId
            });

            bill.paidAmount += payAmount;

            // Check if fully paid
            if (bill.paidAmount >= bill.amount - 0.01) {
                bill.isPaid = true;
            }

            this.saveData();
            return true;
        }
        return false;
    },

    unpayBill(billId) {
        const bill = this.data.bills.find(b => b.id === billId);
        if (bill) {
            // Refund all payments
            if (bill.payments) {
                bill.payments.forEach(p => this.deleteTransaction(p.transactionId));
            } else if (bill.paymentTransactionId) {
                // Legacy single payment support
                this.deleteTransaction(bill.paymentTransactionId);
            }

            bill.payments = [];
            bill.paidAmount = 0;
            bill.isPaid = false;
            bill.paymentTransactionId = null;
            bill.paymentDate = null;

            this.saveData();
            return true;
        }
        return false;
    },

    deleteBill(id) {
        const index = this.data.bills.findIndex(b => b.id === id);
        if (index !== -1) {
            this.data.bills.splice(index, 1);
            this.saveData();
            this.log("Bill deleted", id);
        }
    },

    getPendingBills() {
        return this.data.bills.filter(b => !b.isPaid);
    },

    // --- Clients Management ---
    addClient(clientData) {
        const id = Date.now().toString();
        const client = {
            id,
            name: this.sanitizeInput(clientData.name),
            rfc: this.sanitizeInput(clientData.rfc || ''),
            address: this.sanitizeInput(clientData.address || ''),
            contact: this.sanitizeInput(clientData.contact || ''),
            phone: this.sanitizeInput(clientData.phone || ''),
            email: this.sanitizeInput(clientData.email || ''),
            createdAt: new Date()
        };
        if (!this.data.clients) this.data.clients = [];
        this.data.clients.push(client);
        this.saveData();
        return client;
    },

    updateClient(id, updates) {
        if (!this.data.clients) return false;
        const index = this.data.clients.findIndex(c => c.id === id);
        if (index !== -1) {
            ['name', 'rfc', 'address', 'contact', 'phone', 'email'].forEach(field => {
                if (updates[field]) updates[field] = this.sanitizeInput(updates[field]);
            });
            this.data.clients[index] = { ...this.data.clients[index], ...updates };
            this.saveData();
            return true;
        }
        return false;
    },

    deleteClient(id) {
        if (!this.data.clients) return false;
        const index = this.data.clients.findIndex(c => c.id === id);
        if (index !== -1) {
            this.data.clients.splice(index, 1);
            this.saveData();
            return true;
        }
        return false;
    },

    getClients() {
        return this.data.clients || [];
    },

    // --- Receivables (Cuentas por Cobrar) ---
    addReceivable(data) {
        const id = Date.now().toString();
        const receivable = {
            id,
            number: this.sanitizeInput(data.number) || `A-${Date.now().toString().slice(-6)}`, // Auto-generate simple invoice number
            clientId: data.clientId,
            clientName: this.sanitizeInput(data.clientName), // Store denormalized for ease
            clientRfc: this.sanitizeInput(data.clientRfc),

            issuedDate: data.issuedDate || new Date().toISOString().split('T')[0],
            dueDate: data.dueDate,

            items: data.items || [], // Array of { description, quantity, price, total }
            subtotal: parseFloat(data.subtotal || 0),
            tax: parseFloat(data.tax || 0),
            total: parseFloat(data.total || 0),

            status: 'pending', // pending, paid, overdue
            paidAmount: 0,
            payments: [],

            logo: data.logo || null, // Base64 logo
            notes: this.sanitizeInput(data.notes || '')
        };

        if (!this.data.receivables) this.data.receivables = [];
        this.data.receivables.push(receivable);
        this.saveData();
        return receivable;
    },

    updateReceivable(id, updates) {
        const index = this.data.receivables.findIndex(r => r.id === id);
        if (index !== -1) {
            // Sanitize explicit string fields if present
            ['clientName', 'clientRfc', 'notes', 'number'].forEach(field => {
                if (updates[field]) updates[field] = this.sanitizeInput(updates[field]);
            });
            this.data.receivables[index] = { ...this.data.receivables[index], ...updates };
            this.saveData();
            return true;
        }
        return false;
    },

    deleteReceivable(id) {
        const index = this.data.receivables.findIndex(r => r.id === id);
        if (index !== -1) {
            this.data.receivables.splice(index, 1);
            this.saveData();
        }
    },

    payReceivable(id, amount, accountId, proof = null) {
        const receivable = this.data.receivables.find(r => r.id === id);
        const account = this.data.accounts.find(a => a.id === accountId);
        const payAmount = parseFloat(amount);

        if (receivable && account && payAmount > 0) {
            // 1. Add Income Transaction
            this.addTransaction('income', payAmount, 'Cobranza', `Cobro: ${receivable.clientName}`, accountId);

            // 2. Update Receivable
            receivable.paidAmount = (receivable.paidAmount || 0) + payAmount;
            if (receivable.paidAmount >= receivable.total - 0.01) {
                receivable.status = 'paid';
            }

            receivable.payments.push({
                date: new Date(),
                amount: payAmount,
                accountId,
                proof: proof // Store proof
            });

            this.saveData();
            return true;
        }
        return false;
    },

    deletePayment(invoiceId, paymentIndex) {
        const receivable = this.data.receivables.find(r => r.id === invoiceId);
        if (!receivable || !receivable.payments[paymentIndex]) return false;

        const payment = receivable.payments[paymentIndex];

        // 1. Revert Paid Amount
        receivable.paidAmount = (receivable.paidAmount || 0) - payment.amount;
        if (receivable.paidAmount < 0) receivable.paidAmount = 0;

        // 2. Revert Status
        if (receivable.paidAmount < receivable.total - 0.01) {
            receivable.status = 'pending';
        }

        // 3. Remove Payment
        receivable.payments.splice(paymentIndex, 1);

        // Optional: Revert Transaction logic would go here if we tracked IDs

        this.saveData();
        return true;
    },

    updatePayment(invoiceId, paymentIndex, newDate, newAmount) {
        const receivable = this.data.receivables.find(r => r.id === invoiceId);
        if (!receivable || !receivable.payments[paymentIndex]) return false;

        const payment = receivable.payments[paymentIndex];
        const oldAmount = payment.amount;
        const finalAmount = parseFloat(newAmount);

        // 1. Revert Old Amount
        receivable.paidAmount = (receivable.paidAmount || 0) - oldAmount;

        // 2. Add New Amount
        receivable.paidAmount += finalAmount;

        // 3. Update Payment Record
        payment.date = newDate;
        payment.amount = finalAmount;

        // 4. Update Status
        if (receivable.paidAmount >= receivable.total - 0.01) {
            receivable.status = 'paid';
        } else {
            receivable.status = 'pending';
        }

        this.saveData();
        return true;
    },

    // --- Dashboard Aggregates ---
    getFinancialHealth() {
        const now = new Date();
        const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);

        // Income vs Expenses (This Month)
        const monthlyTransactions = this.data.transactions.filter(t => new Date(t.date) >= startOfMonth);
        const income = monthlyTransactions.filter(t => t.type === 'income').reduce((sum, t) => sum + t.amount, 0);
        const expenses = monthlyTransactions.filter(t => t.type === 'expense').reduce((sum, t) => sum + t.amount, 0);

        // Accounts Receivable stats
        const totalInvoiced = this.data.receivables ? this.data.receivables.reduce((sum, r) => sum + (r.total || 0), 0) : 0;
        const totalCollected = this.data.receivables ? this.data.receivables.reduce((sum, r) => sum + (r.paidAmount || 0), 0) : 0;
        const collectionRatio = totalInvoiced > 0 ? (totalCollected / totalInvoiced) * 100 : 0;

        // Pending Bills
        const pendingBillsAmount = this.data.bills
            .filter(b => !b.isPaid)
            .reduce((sum, b) => sum + (b.amount - (b.paidAmount || 0)), 0);

        return {
            income,
            expenses,
            savings: income - expenses,
            collectionRatio: Math.round(collectionRatio),
            pendingBillsAmount,
            totalBalance: this.getTotalBalance()
        };
    },

    getProjectedCashFlow(days = 30) {
        let balance = this.getTotalBalance();
        const flow = [];
        const today = new Date();

        // Generate daily projection
        for (let i = 0; i < days; i++) {
            const date = new Date(today);
            date.setDate(today.getDate() + i);
            const dateStr = date.toISOString().split('T')[0];

            // 1. Subtract Bills Due
            const billsDue = this.data.bills.filter(b => !b.isPaid && b.dueDate === dateStr);
            const dailyOut = billsDue.reduce((sum, b) => sum + (b.amount - (b.paidAmount || 0)), 0);

            // 2. Add Expect Receivables (Simple projection: assume due date = pay date)
            const receivablesDue = (this.data.receivables || []).filter(r => r.status !== 'paid' && r.dueDate === dateStr);
            const dailyIn = receivablesDue.reduce((sum, r) => sum + ((r.total || 0) - (r.paidAmount || 0)), 0);

            balance = balance - dailyOut + dailyIn;
            flow.push({ date: dateStr, balance, in: dailyIn, out: dailyOut });
        }
        return flow;
    },

    // --- Todos ---
    addTodo(todoData) {
        const isSimple = typeof todoData === 'string';
        const todo = {
            id: Date.now().toString(),
            title: this.sanitizeInput(isSimple ? todoData : todoData.title),
            description: this.sanitizeInput(isSimple ? '' : (todoData.description || '')),
            priority: isSimple ? 'medium' : (todoData.priority || 'medium'),
            dueDate: isSimple ? null : (todoData.dueDate || null),
            dueTime: isSimple ? null : (todoData.dueTime || null),
            recurring: isSimple ? false : (todoData.recurring || false),
            completed: false,
            createdAt: new Date()
        };
        this.data.todos.unshift(todo);
        this.saveData();
        return todo;
    },

    toggleTodo(id) {
        const todo = this.data.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
            this.saveData();
        }
    },

    deleteTodo(id) {
        const index = this.data.todos.findIndex(t => t.id === id);
        if (index !== -1) {
            this.data.todos.splice(index, 1);
            this.saveData();
        }
    },

    // --- Formatting ---
    formatCurrency(amount) {
        return new Intl.NumberFormat('es-MX', { style: 'currency', currency: 'MXN' }).format(amount);
    },

    // --- Events ---
    listeners: [],
    subscribe(callback) {
        this.listeners.push(callback);
    },
    notifyChange() {
        this.listeners.forEach(cb => cb(this.data));
    }
};

// Auto-init
document.addEventListener('DOMContentLoaded', () => {
    FinanceCore.init();
});
