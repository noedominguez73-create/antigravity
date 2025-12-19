/**
 * FinanceAI - Natural Language Parser for Mis Finanzas
 * Interprets voice/text commands and executes FinanceCore actions.
 */

const FinanceAI = {
    parse(text) {
        text = text.toLowerCase();
        console.log("AI Parsing:", text);

        // 0. Extract Amount (Integers or Decimals)
        const amountMatch = text.match(/\d+(\.\d+)?/);
        const amount = amountMatch ? parseFloat(amountMatch[0]) : 0;

        // 1. TAREA / RECORDATORIO (No monetary checking needed first)
        // "Recordar...", "Tarea...", "Nota..." 
        if (text.startsWith('recordar') || text.startsWith('tarea') || text.startsWith('nota')) {
            const task = text.replace(/recordar|tarea|nota/gi, '').trim();
            if (task) {
                FinanceCore.addTodo(task);
                return { type: 'success', message: `Tarea agregada: "${task}"` };
            }
        }

        // 2. DETECT ACCOUNT (Payment Method)
        // Look for "con [AccountName]" e.g. "con efectivo", "con bancomer"
        // Default to first account if not specified
        let accountId = FinanceCore.data.accounts[0]?.id;
        let accountName = FinanceCore.data.accounts[0]?.name;

        const accounts = FinanceCore.data.accounts;
        for (const acc of accounts) {
            // Check if text explicitly mentions the account name
            if (text.includes(acc.name.toLowerCase())) {
                accountId = acc.id;
                accountName = acc.name;
                break; // Use the first matching account found
            }
        }

        // 3. BILLS / PENDING PAYMENTS
        // "Factura...", "Recibo...", OR "...pendiente de pago"
        const isPending = text.includes('pendiente');
        const isExplicitBill = text.includes('factura') || text.includes('recibo');

        if (isExplicitBill || (isPending && amount > 0)) {
            // Clean up text to get a decent name
            let name = text
                .replace(/factura|recibo|pendiente|de|pago|pagar|con|luz|agua|internet/gi, ' ') // Remove keywords
                .replace(/\d+(\.\d+)?/g, '') // Remove amount
                .trim();

            // If name is empty (e.g. "Factura 500"), try to infer category or use generic
            if (name.length < 2) name = "Gasto Pendiente";

            const today = new Date().toISOString().split('T')[0];

            FinanceCore.addBill({
                name: name.charAt(0).toUpperCase() + name.slice(1),
                amount: amount,
                dueDate: today, // Default to today so it appears immediately
                notes: `Generado por voz: "${text}"`
            });

            return { type: 'success', message: `Registrado como pendiente: "${name}" $${amount}` };
        }

        // 4. INCOME
        if (text.includes('ingreso') || text.includes('gané') || text.includes('pagaron') || text.includes('depositaron')) {
            if (amount > 0) {
                FinanceCore.addTransaction('income', amount, 'Salario', text, accountId);
                return { type: 'success', message: `Ingreso de $${amount} registrado en ${accountName}.` };
            }
        }

        // 5. EXPENSE (Default Fallback)
        // If we have an amount, assume it's an expense unless proved otherwise
        if (amount > 0) {
            let category = 'Otros';
            // Try to match category
            const categories = FinanceCore.data.categories;
            for (const cat of categories) {
                if (text.includes(cat.name.toLowerCase())) {
                    category = cat.name;
                    break;
                }
                // Check subcategories
                if (cat.subcategories) {
                    for (const sub of cat.subcategories) {
                        if (text.includes(sub.toLowerCase())) {
                            category = cat.name; // Use parent category
                            break;
                        }
                    }
                }
            }

            const desc = text.charAt(0).toUpperCase() + text.slice(1);
            FinanceCore.addTransaction('expense', amount, category, desc, accountId);

            return { type: 'success', message: `Gasto de $${amount} (${category}) registrado en ${accountName}.` };
        }

        return { type: 'unknown', message: 'No entendí. Prueba: "Super 500 con efectivo" o "Mecánico 2000 pendiente".' };
    }
};
