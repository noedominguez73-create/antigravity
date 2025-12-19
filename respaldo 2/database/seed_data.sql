-- Seed Data for asesoriaimss.io
-- Realistic Mexican data for testing and development

-- ============================================
-- ADMIN USER
-- ============================================
-- Password: Admin123!
INSERT INTO users (email, password_hash, full_name, role, created_at) VALUES
('admin@asesoriaimss.io', 'pbkdf2:sha256:600000$randomsalt$hashedpassword', 'Administrador Sistema', 'admin', NOW());

-- ============================================
-- REGULAR USERS (50 users)
-- ============================================
INSERT INTO users (email, password_hash, full_name, role, created_at) VALUES
('maria.garcia@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'María García López', 'user', NOW() - INTERVAL '45 days'),
('juan.martinez@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Juan Martínez Rodríguez', 'user', NOW() - INTERVAL '40 days'),
('ana.hernandez@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Ana Hernández Pérez', 'user', NOW() - INTERVAL '38 days'),
('carlos.lopez@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Carlos López Sánchez', 'user', NOW() - INTERVAL '35 days'),
('laura.gonzalez@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Laura González Ramírez', 'user', NOW() - INTERVAL '32 days'),
('miguel.rodriguez@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Miguel Rodríguez Torres', 'user', NOW() - INTERVAL '30 days'),
('sofia.sanchez@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Sofía Sánchez Flores', 'user', NOW() - INTERVAL '28 days'),
('diego.ramirez@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Diego Ramírez Cruz', 'user', NOW() - INTERVAL '25 days'),
('valentina.torres@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Valentina Torres Morales', 'user', NOW() - INTERVAL '22 days'),
('fernando.flores@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Fernando Flores Jiménez', 'user', NOW() - INTERVAL '20 days'),
('camila.morales@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Camila Morales Ruiz', 'user', NOW() - INTERVAL '18 days'),
('ricardo.cruz@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Ricardo Cruz Mendoza', 'user', NOW() - INTERVAL '15 days'),
('daniela.jimenez@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Daniela Jiménez Castro', 'user', NOW() - INTERVAL '12 days'),
('alejandro.mendoza@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Alejandro Mendoza Vargas', 'user', NOW() - INTERVAL '10 days'),
('isabella.castro@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Isabella Castro Ortiz', 'user', NOW() - INTERVAL '8 days'),
('roberto.vargas@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Roberto Vargas Reyes', 'user', NOW() - INTERVAL '7 days'),
('gabriela.ortiz@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Gabriela Ortiz Herrera', 'user', NOW() - INTERVAL '6 days'),
('eduardo.reyes@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Eduardo Reyes Medina', 'user', NOW() - INTERVAL '5 days'),
('mariana.herrera@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Mariana Herrera Guzmán', 'user', NOW() - INTERVAL '4 days'),
('javier.medina@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Javier Medina Silva', 'user', NOW() - INTERVAL '3 days'),
('andrea.guzman@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Andrea Guzmán Rojas', 'user', NOW() - INTERVAL '2 days'),
('pablo.silva@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Pablo Silva Navarro', 'user', NOW() - INTERVAL '1 day'),
('natalia.rojas@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Natalia Rojas Cortés', 'user', NOW()),
('sergio.navarro@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Sergio Navarro Delgado', 'user', NOW()),
('carolina.cortes@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Carolina Cortés Aguilar', 'user', NOW()),
('antonio.delgado@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Antonio Delgado Vega', 'user', NOW()),
('valeria.aguilar@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Valeria Aguilar Romero', 'user', NOW()),
('manuel.vega@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Manuel Vega Castillo', 'user', NOW()),
('luciana.romero@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Luciana Romero Ramos', 'user', NOW()),
('francisco.castillo@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Francisco Castillo Fuentes', 'user', NOW()),
('renata.ramos@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Renata Ramos Molina', 'user', NOW()),
('gustavo.fuentes@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Gustavo Fuentes Paredes', 'user', NOW()),
('emilia.molina@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Emilia Molina Salazar', 'user', NOW()),
('raul.paredes@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Raúl Paredes Domínguez', 'user', NOW()),
('martina.salazar@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Martina Salazar Ibarra', 'user', NOW()),
('adrian.dominguez@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Adrián Domínguez Lara', 'user', NOW()),
('victoria.ibarra@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Victoria Ibarra Campos', 'user', NOW()),
('jorge.lara@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Jorge Lara Pacheco', 'user', NOW()),
('regina.campos@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Regina Campos Ríos', 'user', NOW()),
('hector.pacheco@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Héctor Pacheco Guerrero', 'user', NOW()),
('ximena.rios@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Ximena Ríos Núñez', 'user', NOW()),
('cesar.guerrero@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'César Guerrero Peña', 'user', NOW()),
('paula.nunez@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Paula Núñez Soto', 'user', NOW()),
('oscar.pena@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Óscar Peña Márquez', 'user', NOW()),
('juliana.soto@yahoo.com', 'pbkdf2:sha256:600000$salt$hash', 'Juliana Soto Blanco', 'user', NOW()),
('luis.marquez@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Luis Márquez Alvarado', 'user', NOW()),
('catalina.blanco@hotmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Catalina Blanco Carrillo', 'user', NOW()),
('ignacio.alvarado@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Ignacio Alvarado Espinoza', 'user', NOW()),
('miranda.carrillo@outlook.com', 'pbkdf2:sha256:600000$salt$hash', 'Miranda Carrillo Valdez', 'user', NOW()),
('alberto.espinoza@gmail.com', 'pbkdf2:sha256:600000$salt$hash', 'Alberto Espinoza Montes', 'user', NOW());

-- ============================================
-- PROFESSIONAL USERS (150 professionals)
-- ============================================
-- Specialties: Abogado, Contador, Arquitecto, Médico, Psicólogo, Nutriólogo, Dentista, Ingeniero Civil,
--              Ingeniero Industrial, Diseñador Gráfico, Desarrollador Web, Marketing Digital, Fotógrafo,
--              Veterinario, Electricista, Plomero, Mecánico, Estilista, Chef, Personal Trainer, Maestro, Consultor

-- Note: This is a condensed version. In production, you would generate all 150.
-- Here's a representative sample across specialties and cities:

INSERT INTO users (email, password_hash, full_name, role, created_at) VALUES
-- Abogados
('lic.ramirez@legal.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Roberto Ramírez Soto', 'professional', NOW() - INTERVAL '120 days'),
('lic.mendoza@juridico.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Patricia Mendoza Luna', 'professional', NOW() - INTERVAL '115 days'),
('lic.torres@abogados.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Fernando Torres Vega', 'professional', NOW() - INTERVAL '110 days'),
('lic.garcia@derecho.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Carmen García Ruiz', 'professional', NOW() - INTERVAL '105 days'),
('lic.lopez@legal.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Miguel López Castro', 'professional', NOW() - INTERVAL '100 days'),
('lic.hernandez@juridico.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Ana Hernández Morales', 'professional', NOW() - INTERVAL '95 days'),
('lic.martinez@abogados.mx', 'pbkdf2:sha256:600000$salt$hash', 'Lic. Jorge Martínez Flores', 'professional', NOW() - INTERVAL '90 days'),

-- Contadores
('cp.sanchez@contadores.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. Laura Sánchez Ortiz', 'professional', NOW() - INTERVAL '85 days'),
('cp.rodriguez@fiscal.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. Carlos Rodríguez Pérez', 'professional', NOW() - INTERVAL '80 days'),
('cp.gonzalez@contabilidad.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. María González Reyes', 'professional', NOW() - INTERVAL '75 days'),
('cp.diaz@contadores.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. Ricardo Díaz Herrera', 'professional', NOW() - INTERVAL '70 days'),
('cp.cruz@fiscal.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. Sofía Cruz Medina', 'professional', NOW() - INTERVAL '65 days'),
('cp.morales@contabilidad.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. Antonio Morales Guzmán', 'professional', NOW() - INTERVAL '60 days'),
('cp.jimenez@contadores.mx', 'pbkdf2:sha256:600000$salt$hash', 'CP. Gabriela Jiménez Silva', 'professional', NOW() - INTERVAL '55 days'),

-- Arquitectos
('arq.vargas@arquitectura.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Diego Vargas Rojas', 'professional', NOW() - INTERVAL '50 days'),
('arq.castillo@diseno.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Valentina Castillo Navarro', 'professional', NOW() - INTERVAL '48 days'),
('arq.ramos@arquitectura.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Eduardo Ramos Cortés', 'professional', NOW() - INTERVAL '46 days'),
('arq.fuentes@diseno.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Isabella Fuentes Delgado', 'professional', NOW() - INTERVAL '44 days'),
('arq.molina@arquitectura.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Roberto Molina Aguilar', 'professional', NOW() - INTERVAL '42 days'),
('arq.paredes@diseno.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Camila Paredes Vega', 'professional', NOW() - INTERVAL '40 days'),
('arq.salazar@arquitectura.mx', 'pbkdf2:sha256:600000$salt$hash', 'Arq. Fernando Salazar Romero', 'professional', NOW() - INTERVAL '38 days');

-- Continue with more professionals... (condensed for brevity)
-- In production, generate all 150 across 22 specialties and 30 cities

-- ============================================
-- PROFESSIONALS TABLE (matching users above)
-- ============================================
INSERT INTO professionals (user_id, specialty, city, bio, rating, total_reviews, is_active, created_at) VALUES
-- Abogados
(52, 'Abogado', 'Ciudad de México', 'Especialista en derecho laboral con 15 años de experiencia. Asesoría en trámites IMSS, pensiones y seguridad social.', 4.8, 24, TRUE, NOW() - INTERVAL '120 days'),
(53, 'Abogado', 'Guadalajara', 'Abogada experta en derecho familiar y civil. Más de 10 años defendiendo los derechos de trabajadores.', 4.6, 18, TRUE, NOW() - INTERVAL '115 days'),
(54, 'Abogado', 'Monterrey', 'Litigante especializado en seguridad social y prestaciones laborales. Atención personalizada.', 4.9, 31, TRUE, NOW() - INTERVAL '110 days'),
(55, 'Abogado', 'Puebla', 'Asesoría legal integral en trámites IMSS, incapacidades y pensiones. Consulta gratuita.', 4.7, 22, TRUE, NOW() - INTERVAL '105 days'),
(56, 'Abogado', 'Tijuana', 'Experto en derecho laboral y seguridad social. Casos ganados en IMSS y ISSSTE.', 4.5, 15, TRUE, NOW() - INTERVAL '100 days'),
(57, 'Abogado', 'León', 'Abogada con maestría en derecho laboral. Especialista en conflictos con el IMSS.', 4.8, 27, TRUE, NOW() - INTERVAL '95 days'),
(58, 'Abogado', 'Querétaro', 'Asesoría legal en pensiones, incapacidades y trámites ante el IMSS. 12 años de experiencia.', 4.6, 19, TRUE, NOW() - INTERVAL '90 days'),

-- Contadores
(59, 'Contador', 'Ciudad de México', 'Contador público certificado. Especialista en nóminas, IMSS e INFONAVIT. Asesoría fiscal integral.', 4.9, 35, TRUE, NOW() - INTERVAL '85 days'),
(60, 'Contador', 'Guadalajara', 'Experta en cálculo de cuotas obrero-patronales. Auditorías IMSS y regularización de trabajadores.', 4.7, 28, TRUE, NOW() - INTERVAL '80 days'),
(61, 'Contador', 'Monterrey', 'Contador con 20 años de experiencia. Trámites IMSS, altas, bajas y modificaciones salariales.', 4.8, 32, TRUE, NOW() - INTERVAL '75 days'),
(62, 'Contador', 'Mérida', 'Asesoría contable y fiscal para empresas. Especialista en obligaciones patronales ante IMSS.', 4.6, 21, TRUE, NOW() - INTERVAL '70 days'),
(63, 'Contador', 'Cancún', 'Contadora pública con maestría en fiscal. Cálculo de finiquitos y liquidaciones conforme a ley.', 4.9, 29, TRUE, NOW() - INTERVAL '65 days'),
(64, 'Contador', 'Aguascalientes', 'Experto en nóminas y seguridad social. Regularización de trabajadores ante IMSS e INFONAVIT.', 4.5, 17, TRUE, NOW() - INTERVAL '60 days'),
(65, 'Contador', 'Toluca', 'Contadora especializada en auditorías IMSS. Defensa en revisiones y multas patronales.', 4.7, 23, TRUE, NOW() - INTERVAL '55 days'),

-- Arquitectos
(66, 'Arquitecto', 'Ciudad de México', 'Arquitecto con 12 años de experiencia en diseño residencial y comercial. Proyectos integrales.', 4.8, 26, TRUE, NOW() - INTERVAL '50 days'),
(67, 'Arquitecto', 'Guadalajara', 'Diseño arquitectónico sustentable. Especialista en remodelaciones y ampliaciones.', 4.6, 20, TRUE, NOW() - INTERVAL '48 days'),
(68, 'Arquitecto', 'Monterrey', 'Arquitecto con maestría en diseño urbano. Proyectos residenciales de lujo.', 4.9, 33, TRUE, NOW() - INTERVAL '46 days'),
(69, 'Arquitecto', 'Playa del Carmen', 'Diseño de espacios comerciales y turísticos. Más de 50 proyectos realizados.', 4.7, 24, TRUE, NOW() - INTERVAL '44 days'),
(70, 'Arquitecto', 'San Luis Potosí', 'Arquitecto especializado en diseño industrial y corporativo. Atención personalizada.', 4.5, 16, TRUE, NOW() - INTERVAL '42 days'),
(71, 'Arquitecto', 'Morelia', 'Diseño arquitectónico contemporáneo. Proyectos residenciales y comerciales llave en mano.', 4.8, 28, TRUE, NOW() - INTERVAL '40 days'),
(72, 'Arquitecto', 'Cuernavaca', 'Arquitecta con 15 años de experiencia. Especialista en casas de campo y residencias.', 4.6, 22, TRUE, NOW() - INTERVAL '38 days');

-- Note: In production, continue with all 150 professionals across all specialties

-- ============================================
-- SERVICES (3-5 per professional)
-- ============================================
INSERT INTO services (professional_id, name, description, price, duration_minutes, created_at) VALUES
-- Services for Abogados
(1, 'Consulta Legal Inicial', 'Primera consulta para evaluar tu caso y definir estrategia legal', 500.00, 60, NOW()),
(1, 'Asesoría en Pensiones IMSS', 'Trámite completo de pensión por cesantía, vejez o invalidez', 3500.00, 120, NOW()),
(1, 'Defensa en Incapacidades', 'Representación legal en casos de incapacidad permanente o temporal', 5000.00, 180, NOW()),
(1, 'Trámite de Pensión Alimenticia', 'Gestión legal de pensión alimenticia ante autoridades', 4000.00, 90, NOW()),

(2, 'Consulta Jurídica', 'Asesoría legal personalizada en derecho laboral y familiar', 600.00, 60, NOW()),
(2, 'Divorcio Express', 'Trámite de divorcio voluntario o necesario', 8000.00, 240, NOW()),
(2, 'Demanda Laboral', 'Representación en juicios laborales ante Junta de Conciliación', 10000.00, 300, NOW()),

(3, 'Asesoría Legal IMSS', 'Consultoría especializada en trámites y prestaciones IMSS', 800.00, 90, NOW()),
(3, 'Pensión por Invalidez', 'Gestión completa de pensión por invalidez permanente', 6000.00, 150, NOW()),
(3, 'Recurso de Inconformidad', 'Defensa legal en negativas de prestaciones IMSS', 7500.00, 180, NOW()),
(3, 'Cálculo de Semanas Cotizadas', 'Verificación y corrección de semanas cotizadas ante IMSS', 2500.00, 60, NOW()),

-- Services for Contadores
(8, 'Cálculo de Nómina', 'Procesamiento de nómina mensual con cálculo de IMSS e impuestos', 2000.00, 120, NOW()),
(8, 'Alta de Trabajadores IMSS', 'Registro de nuevos empleados ante IMSS e INFONAVIT', 500.00, 30, NOW()),
(8, 'Auditoría IMSS', 'Revisión preventiva de obligaciones patronales ante IMSS', 8000.00, 240, NOW()),
(8, 'Declaración Anual', 'Presentación de declaración anual de personas físicas', 3000.00, 90, NOW()),

(9, 'Asesoría Fiscal', 'Consultoría fiscal y contable para empresas', 1500.00, 90, NOW()),
(9, 'Cálculo de Finiquito', 'Determinación de finiquito o liquidación conforme a ley', 800.00, 60, NOW()),
(9, 'Regularización IMSS', 'Corrección de diferencias en cuotas obrero-patronales', 5000.00, 180, NOW()),

-- Services for Arquitectos
(15, 'Diseño Arquitectónico', 'Proyecto arquitectónico completo para casa habitación', 25000.00, 480, NOW()),
(15, 'Remodelación', 'Diseño y supervisión de remodelación de espacios', 15000.00, 360, NOW()),
(15, 'Planos Arquitectónicos', 'Elaboración de planos para trámites municipales', 8000.00, 240, NOW()),
(15, 'Asesoría en Construcción', 'Consultoría técnica durante proceso constructivo', 2000.00, 120, NOW()),

(16, 'Proyecto Residencial', 'Diseño integral de vivienda unifamiliar', 30000.00, 600, NOW()),
(16, 'Diseño de Interiores', 'Proyecto de interiorismo y decoración', 12000.00, 300, NOW()),
(16, 'Renders 3D', 'Visualización fotorrealística del proyecto', 5000.00, 180, NOW());

-- Continue with services for all professionals...

-- ============================================
-- EXPERIENCES (2-3 per professional)
-- ============================================
INSERT INTO experiences (professional_id, company, position, description, start_date, end_date, is_current, created_at) VALUES
-- Experiences for Abogados
(1, 'Bufete Jurídico Ramírez & Asociados', 'Socio Fundador', 'Dirección de área laboral y seguridad social. Atención de más de 500 casos exitosos.', '2015-01-01', NULL, TRUE, NOW()),
(1, 'Despacho Legal Corporativo', 'Abogado Senior', 'Litigio laboral y asesoría en prestaciones de ley. Especialización en IMSS.', '2010-03-01', '2014-12-31', FALSE, NOW()),
(1, 'Junta Federal de Conciliación y Arbitraje', 'Asesor Jurídico', 'Asesoría a trabajadores en conflictos laborales y trámites de seguridad social.', '2008-06-01', '2010-02-28', FALSE, NOW()),

(2, 'Despacho Mendoza Abogados', 'Directora Legal', 'Especialización en derecho familiar y laboral. Casos de pensiones y divorcios.', '2018-01-01', NULL, TRUE, NOW()),
(2, 'Tribunal Superior de Justicia', 'Secretaria de Acuerdos', 'Gestión de expedientes en materia familiar y civil.', '2013-08-01', '2017-12-31', FALSE, NOW()),

-- Experiences for Contadores
(8, 'Despacho Contable Sánchez', 'Contador Público Certificado', 'Asesoría fiscal y contable a empresas. Especialista en nóminas y IMSS.', '2016-01-01', NULL, TRUE, NOW()),
(8, 'Corporativo Industrial del Norte', 'Gerente de Nóminas', 'Administración de nómina para 500+ empleados. Cálculo de IMSS e INFONAVIT.', '2012-03-01', '2015-12-31', FALSE, NOW()),
(8, 'KPMG México', 'Auditor Senior', 'Auditorías fiscales y revisiones de cumplimiento IMSS para clientes corporativos.', '2009-07-01', '2012-02-28', FALSE, NOW()),

-- Experiences for Arquitectos
(15, 'Estudio de Arquitectura Vargas', 'Arquitecto Principal', 'Diseño y dirección de proyectos residenciales y comerciales de alto nivel.', '2017-01-01', NULL, TRUE, NOW()),
(15, 'Constructora Inmobiliaria del Bajío', 'Arquitecto de Proyecto', 'Desarrollo de proyectos habitacionales y fraccionamientos.', '2013-05-01', '2016-12-31', FALSE, NOW());

-- Continue with experiences for all professionals...

-- ============================================
-- CERTIFICATIONS (1-2 per professional)
-- ============================================
INSERT INTO certifications (professional_id, name, issuer, issue_date, expiry_date, credential_id, credential_url, created_at) VALUES
-- Certifications for professionals
(1, 'Especialidad en Derecho Laboral', 'Universidad Nacional Autónoma de México', '2015-06-15', NULL, 'UNAM-DL-2015-1234', 'https://certificados.unam.mx/1234', NOW()),
(1, 'Diplomado en Seguridad Social', 'Instituto Mexicano del Seguro Social', '2018-11-20', '2023-11-20', 'IMSS-DSS-2018-5678', NULL, NOW()),

(2, 'Maestría en Derecho Familiar', 'Universidad de Guadalajara', '2016-08-10', NULL, 'UDG-MDF-2016-9012', 'https://certificados.udg.mx/9012', NOW()),

(8, 'Contador Público Certificado', 'Instituto Mexicano de Contadores Públicos', '2014-05-20', '2024-05-20', 'IMCP-CPC-2014-3456', 'https://imcp.org.mx/cert/3456', NOW()),
(8, 'Certificación en Nóminas y Seguridad Social', 'AMCP', '2019-03-15', '2024-03-15', 'AMCP-NSS-2019-7890', NULL, NOW()),

(15, 'Arquitecto Certificado', 'Federación de Colegios de Arquitectos de la República Mexicana', '2013-09-10', '2023-09-10', 'FCARM-AC-2013-2345', 'https://fcarm.mx/cert/2345', NOW()),
(15, 'Diplomado en Diseño Sustentable', 'UNAM', '2020-01-25', NULL, 'UNAM-DS-2020-6789', NULL, NOW());

-- Continue with certifications...

-- ============================================
-- CATEGORIES
-- ============================================
INSERT INTO categories (name, description) VALUES
('Trámites IMSS', 'Consultas sobre trámites y servicios del IMSS'),
('Pensiones', 'Información sobre pensiones y retiro'),
('Incapacidades', 'Dudas sobre incapacidades médicas'),
('Servicios Médicos', 'Consultas sobre atención médica IMSS'),
('Afiliación', 'Preguntas sobre afiliación y registro'),
('Prestaciones', 'Información sobre prestaciones y beneficios');

-- ============================================
-- COMMENTS (300 comments distributed across professionals)
-- ============================================
-- Note: Condensed version showing representative samples

INSERT INTO comments (professional_id, user_id, rating, content, status, created_at) VALUES
-- Comments for Abogado #1
(1, 2, 5, 'Excelente abogado, me ayudó con mi pensión del IMSS. Muy profesional y atento.', 'approved', NOW() - INTERVAL '10 days'),
(1, 3, 5, 'Recomendado 100%. Resolvió mi caso de incapacidad permanente de manera exitosa.', 'approved', NOW() - INTERVAL '8 days'),
(1, 4, 4, 'Buen servicio, aunque tardó un poco más de lo esperado. Pero al final todo salió bien.', 'approved', NOW() - INTERVAL '5 days'),
(1, 5, 5, 'El Lic. Ramírez es muy conocedor del tema. Me orientó perfectamente en mi trámite.', 'approved', NOW() - INTERVAL '3 days'),

-- Comments for Abogado #2
(2, 6, 5, 'La Lic. Mendoza me ayudó con mi divorcio. Muy profesional y empática.', 'approved', NOW() - INTERVAL '12 days'),
(2, 7, 4, 'Buena atención, explicó todo el proceso claramente. Precios justos.', 'approved', NOW() - INTERVAL '9 days'),
(2, 8, 5, 'Excelente abogada, ganamos el caso laboral. Totalmente recomendada.', 'approved', NOW() - INTERVAL '6 days'),

-- Comments for Abogado #3
(3, 9, 5, 'El mejor abogado en temas de IMSS. Me ayudó a recuperar mis semanas cotizadas.', 'approved', NOW() - INTERVAL '15 days'),
(3, 10, 5, 'Muy profesional y honesto. Explica todo con claridad y paciencia.', 'approved', NOW() - INTERVAL '11 days'),
(3, 11, 4, 'Buen servicio, resolvió mis dudas sobre pensión. Recomendado.', 'approved', NOW() - INTERVAL '7 days'),
(3, 12, 5, 'Excelente asesoría legal. Logró que me aprobaran mi pensión por invalidez.', 'approved', NOW() - INTERVAL '4 days'),

-- Comments for Contador #8
(8, 13, 5, 'La CP Sánchez lleva la nómina de mi empresa. Muy profesional y puntual.', 'approved', NOW() - INTERVAL '14 days'),
(8, 14, 5, 'Excelente servicio contable. Nos ayudó a regularizar nuestras obligaciones con el IMSS.', 'approved', NOW() - INTERVAL '10 days'),
(8, 15, 4, 'Buena contadora, conoce muy bien los temas de IMSS e INFONAVIT.', 'approved', NOW() - INTERVAL '6 days'),

-- Comments for Contador #9
(9, 16, 5, 'El CP Rodríguez es muy eficiente. Nos hizo la auditoría IMSS sin problemas.', 'approved', NOW() - INTERVAL '13 days'),
(9, 17, 5, 'Excelente asesoría fiscal. Nos ahorró mucho dinero en multas del IMSS.', 'approved', NOW() - INTERVAL '9 days'),
(9, 18, 4, 'Buen contador, muy profesional. Precios competitivos.', 'approved', NOW() - INTERVAL '5 days'),

-- Comments for Arquitecto #15
(15, 19, 5, 'El Arq. Vargas diseñó mi casa. Quedó espectacular, superó mis expectativas.', 'approved', NOW() - INTERVAL '20 days'),
(15, 20, 5, 'Excelente arquitecto, muy creativo y profesional. Cumplió con los tiempos.', 'approved', NOW() - INTERVAL '16 days'),
(15, 21, 4, 'Buen trabajo, aunque hubo algunos cambios en el proyecto original.', 'approved', NOW() - INTERVAL '12 days'),

-- Comments for Arquitecto #16
(16, 22, 5, 'La Arq. Castillo tiene un estilo único. Mi casa quedó hermosa.', 'approved', NOW() - INTERVAL '18 days'),
(16, 23, 5, 'Muy profesional y atenta. Los renders fueron impresionantes.', 'approved', NOW() - INTERVAL '14 days'),
(16, 24, 5, 'Recomendada al 100%. Excelente diseño de interiores.', 'approved', NOW() - INTERVAL '10 days');

-- Continue with 300 total comments...

-- ============================================
-- CHATBOT CONFIGURATIONS
-- ============================================
INSERT INTO chatbot_configs (professional_id, is_active, welcome_message, system_prompt, knowledge_base, max_tokens, temperature, created_at) VALUES
(1, TRUE, '¡Hola! Soy el asistente virtual del Lic. Roberto Ramírez. ¿En qué puedo ayudarte hoy con tus trámites del IMSS?',
'Eres un asistente legal especializado en derecho laboral y seguridad social en México. Ayudas a las personas con información sobre trámites del IMSS, pensiones, incapacidades y prestaciones laborales. Siempre recomienda agendar una consulta con el abogado para casos específicos.',
'Servicios: Pensiones IMSS, Incapacidades, Trámites de Seguridad Social. Horario: Lunes a Viernes 9am-6pm. Ubicación: Ciudad de México.',
1000, 0.7, NOW()),

(8, TRUE, '¡Bienvenido! Soy el asistente de la CP Laura Sánchez. ¿Tienes dudas sobre nóminas o trámites IMSS?',
'Eres un asistente contable especializado en nóminas, IMSS e INFONAVIT. Ayudas con información sobre cálculos de cuotas, altas y bajas de trabajadores, y obligaciones patronales. Recomienda agendar consulta para asesoría personalizada.',
'Servicios: Nóminas, Altas IMSS, Auditorías, Declaraciones. Experiencia: 15 años en contabilidad y fiscal. Ubicación: Ciudad de México.',
1000, 0.7, NOW()),

(15, TRUE, 'Hola, soy el asistente del Arq. Diego Vargas. ¿Interesado en diseño arquitectónico?',
'Eres un asistente de arquitectura que ayuda con información sobre proyectos arquitectónicos, remodelaciones y diseño. Proporcionas información general sobre servicios y costos aproximados. Siempre recomienda agendar una cita para cotización personalizada.',
'Servicios: Diseño Arquitectónico, Remodelaciones, Planos, Renders 3D. Especialidad: Proyectos Residenciales. Ubicación: Ciudad de México.',
1000, 0.7, NOW());

-- ============================================
-- CHAT MESSAGES (Sample conversations)
-- ============================================
INSERT INTO chat_messages (professional_id, user_id, session_id, role, content, credits_used, created_at) VALUES
-- Conversation 1 with Abogado
(1, 2, 'session-001', 'user', '¿Cómo puedo tramitar mi pensión del IMSS?', 0, NOW() - INTERVAL '5 days'),
(1, 2, 'session-001', 'assistant', 'Para tramitar tu pensión del IMSS necesitas cumplir con ciertos requisitos dependiendo del tipo de pensión. ¿Es pensión por cesantía en edad avanzada, vejez o invalidez? El Lic. Ramírez puede asesorarte personalmente. ¿Te gustaría agendar una consulta?', 1, NOW() - INTERVAL '5 days'),
(1, 2, 'session-001', 'user', 'Es por cesantía, tengo 62 años', 0, NOW() - INTERVAL '5 days'),
(1, 2, 'session-001', 'assistant', 'Para pensión por cesantía necesitas tener al menos 60 años (✓) y 1,250 semanas cotizadas. El Lic. Ramírez puede verificar tus semanas y ayudarte con todo el trámite. El costo de la asesoría completa es de $3,500 MXN. ¿Deseas agendar una cita?', 1, NOW() - INTERVAL '5 days'),

-- Conversation 2 with Contador
(8, 13, 'session-002', 'user', '¿Cuánto cuesta el servicio de nómina mensual?', 0, NOW() - INTERVAL '3 days'),
(8, 13, 'session-002', 'assistant', 'El servicio de cálculo de nómina mensual tiene un costo de $2,000 MXN e incluye el procesamiento completo con cálculo de IMSS, INFONAVIT e impuestos. ¿Cuántos empleados tienes en tu empresa?', 1, NOW() - INTERVAL '3 days'),
(8, 13, 'session-002', 'user', 'Tengo 15 empleados', 0, NOW() - INTERVAL '3 days'),
(8, 13, 'session-002', 'assistant', 'Perfecto, para 15 empleados el servicio mensual de nómina es de $2,000 MXN. La CP Sánchez puede ofrecerte una cotización personalizada y explicarte todo el proceso. ¿Te gustaría agendar una consulta?', 1, NOW() - INTERVAL '3 days'),

-- Conversation 3 with Arquitecto
(15, 19, 'session-003', 'user', '¿Cuánto cuesta un proyecto arquitectónico?', 0, NOW() - INTERVAL '7 days'),
(15, 19, 'session-003', 'assistant', 'El costo de un proyecto arquitectónico completo para casa habitación es de $25,000 MXN e incluye planos arquitectónicos, estructurales y de instalaciones. El precio puede variar según el tamaño y complejidad. ¿De cuántos m² sería tu proyecto?', 1, NOW() - INTERVAL '7 days'),
(15, 19, 'session-003', 'user', 'Aproximadamente 150 m²', 0, NOW() - INTERVAL '7 days'),
(15, 19, 'session-003', 'assistant', 'Para una casa de 150 m² el proyecto completo estaría en $25,000 - $30,000 MXN. El Arq. Vargas puede darte una cotización exacta después de conocer tus necesidades específicas. ¿Deseas agendar una cita para platicar tu proyecto?', 1, NOW() - INTERVAL '7 days');

-- ============================================
-- CREDITS (Sample credit purchases)
-- ============================================
INSERT INTO credits (professional_id, transaction_type, transaction_amount, payment_method, payment_status, price_mxn, created_at) VALUES
(1, 'purchase', 100, 'clabe', 'confirmed', 30.00, NOW() - INTERVAL '30 days'),
(1, 'purchase', 50, 'oxxo', 'confirmed', 15.00, NOW() - INTERVAL '15 days'),
(8, 'purchase', 200, 'clabe', 'confirmed', 60.00, NOW() - INTERVAL '25 days'),
(8, 'purchase', 100, 'efectivo', 'confirmed', 30.00, NOW() - INTERVAL '10 days'),
(15, 'purchase', 150, 'clabe', 'confirmed', 45.00, NOW() - INTERVAL '20 days'),
(15, 'purchase', 75, 'oxxo', 'pending', 22.50, NOW() - INTERVAL '2 days');

-- ============================================
-- REFERRALS (Sample referral codes)
-- ============================================
INSERT INTO referrals (referrer_id, referred_user_id, referral_code, commission_rate, total_earned_mxn, status, expires_at, created_at) VALUES
(1, 25, 'RAMIREZ01', 0.20, 6.00, 'active', NOW() + INTERVAL '12 months', NOW() - INTERVAL '60 days'),
(8, 30, 'SANCHEZ01', 0.20, 12.00, 'active', NOW() + INTERVAL '12 months', NOW() - INTERVAL '45 days'),
(15, 35, 'VARGAS01', 0.20, 9.00, 'active', NOW() + INTERVAL '12 months', NOW() - INTERVAL '30 days');

-- ============================================
-- INQUIRIES (Sample general inquiries)
-- ============================================
INSERT INTO inquiries (user_id, category_id, subject, message, status, created_at) VALUES
(2, 1, 'Duda sobre trámite de pensión', '¿Cuáles son los requisitos para pensión por cesantía?', 'answered', NOW() - INTERVAL '20 days'),
(5, 3, 'Incapacidad por maternidad', '¿Cuánto tiempo dura la incapacidad por maternidad?', 'answered', NOW() - INTERVAL '15 days'),
(10, 2, 'Cálculo de pensión', '¿Cómo se calcula el monto de mi pensión?', 'pending', NOW() - INTERVAL '5 days'),
(15, 4, 'Citas médicas IMSS', '¿Cómo puedo agendar una cita con especialista?', 'answered', NOW() - INTERVAL '10 days');

-- ============================================
-- RESPONSES (Sample responses to inquiries)
-- ============================================
INSERT INTO responses (inquiry_id, user_id, message, created_at) VALUES
(1, 1, 'Para pensión por cesantía necesitas tener al menos 60 años y 1,250 semanas cotizadas. Te recomendamos contactar a un abogado especializado para asesoría personalizada.', NOW() - INTERVAL '19 days'),
(2, 1, 'La incapacidad por maternidad es de 84 días: 42 días antes del parto y 42 días después. Durante este periodo recibes el 100% de tu salario registrado.', NOW() - INTERVAL '14 days'),
(4, 1, 'Puedes agendar cita con especialista a través de la app IMSS Digital o llamando al 800 623 2323. Necesitas tu número de seguridad social.', NOW() - INTERVAL '9 days');

-- ============================================
-- END OF SEED DATA
-- ============================================

-- Note: This is a condensed version showing the structure and representative samples.
-- In production, you would generate all 150 professionals across 22 specialties and 30 cities,
-- along with their complete services, experiences, certifications, and 300 comments.
